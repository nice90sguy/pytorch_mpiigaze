"""
Use the output of gaze_estimation to calibrate (train) screen_gaze
"""
import logging
import math
import time
from random import random
from typing import List, Tuple
import screeninfo
import cv2
import numpy as np
import torch.nn
import yacs.config

from gaze2screen_model import Gaze2Screen
from gaze_estimation import GazeEstimator
from gaze_estimation.gaze_estimator.common import FacePartsName, Face
from gaze_estimation.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScreenGaze:

    def __init__(self, config: yacs.config.CfgNode):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)

        self.training = False
        self.gaze2screen = Gaze2Screen(self.config.screen_gaze.save_file_name)
        self.loss_func = torch.nn.MSELoss(reduction='sum')
        self.loss = 0
        self.optimizer = torch.optim.SGD(self.gaze2screen.parameters(), lr=1e-3, dampening=.5, momentum=.5)
        self.cap = self._create_capture()
        self.minibatch_size = 100

        self.target = np.ndarray(2)  # x, y coordinate of target, range 0..1
        self.target[0] = 0
        self.target[1] = 0
        self.fade_counter = self.config.screen_gaze.fade_counter
        self.guess = np.zeros(2)
        # get the size of the screen
        self.screen = screeninfo.get_monitors()[self.config.screen_gaze.screen_id]
        self.init_display()

    @staticmethod
    def _convert_pt(point: np.ndarray) -> Tuple[int, int]:
        return tuple(np.round(point).astype(int).tolist())

    def update_label(self):

        w = self.screen.width
        h = self.screen.height

        n_vertical_divisions = 5
        n_horizontal_divisions = 3

        self.fade_counter = self.fade_counter - 1
        if self.fade_counter <= 0:
            self.fade_counter = self.config.screen_gaze.fade_counter
            # Choose a new random position for the target
            self.target[0] = int(n_vertical_divisions * random()) / n_vertical_divisions - 0.5 + 1 / (2 * n_vertical_divisions)
            self.target[1] = int(n_horizontal_divisions * random()) / n_horizontal_divisions - 0.5 + 1 / (2 * n_horizontal_divisions)

        target_pt = self.target.copy()
        target_pt += 0.5   # change from zero to screen-coords
        target_pt[0] *= w
        target_pt[1] *= h
        target_pt = self._convert_pt(target_pt)

        guess_pt = self.guess.copy()

        guess_pt += 0.5
        guess_pt[0] *= w
        guess_pt[1] *= h
        guess_pt = self._convert_pt(guess_pt)
        image = np.zeros((h, w, 3), dtype=np.float32)
        # Draw a dot centred at target coords
        is_in_blanking_period = self.config.screen_gaze.fade_counter - self.fade_counter < self.config.screen_gaze.blanking_period
        if self.training:
            target_brightness_red = self.fade_counter / self.config.screen_gaze.fade_counter + 0.2

            target_brightness_green = .7 if is_in_blanking_period else 0
            cv2.circle(image, target_pt, 100, (0, target_brightness_green, target_brightness_red), cv2.FILLED)

        if not self.training or not is_in_blanking_period:
            cv2.circle(image, guess_pt, 20, (.3, .3, 0), cv2.FILLED)

        for x in range(0, w+1, w // n_vertical_divisions):
            cv2.line(image, (x, 0), (x, h), (1, 0, 0))
        for x in range(0, h+1, h // n_horizontal_divisions):
            cv2.line(image, (0, x), (w, x), (1, 0, 0))
        cv2.imshow(self.config.screen_gaze.window_name, image)







    def term_display(self):
        cv2.destroyAllWindows()

    def init_display(self) -> None:
        # https://gist.github.com/ronekko/dc3747211543165108b11073f929b85e
        width = self.screen.width
        height = self.screen.height

        window_name = self.config.screen_gaze.window_name
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, self.screen.x - 1, self.screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def _create_capture(self) -> cv2.VideoCapture:

        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def train(self) -> None:
        ok = True
        iters = 0
        while ok:
            batch_start_time = time.perf_counter()
            for i in range(self.minibatch_size):

                self.update_label()
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    ok = False
                elif key == ord('t'):
                    self.training = not self.training
                elif key == ord(' '):
                    self.training = not self.training
                    self.fade_counter = 0

                if ok:
                    ok, frame = self.cap.read()
                if not ok:
                    break


                undistorted = cv2.undistort(
                    frame, self.gaze_estimator.camera.camera_matrix,
                    self.gaze_estimator.camera.dist_coefficients)

                faces = self.gaze_estimator.detect_faces(undistorted)
                for face in faces:
                    features_np = self._extract_features(undistorted, face)
                    features = torch.tensor(features_np, dtype=torch.float32)
                    guess = self.gaze2screen(features)
                    self.guess = guess.detach().numpy()
                    if self.training:
                        is_in_blanking_period = self.config.screen_gaze.fade_counter - self.fade_counter < self.config.screen_gaze.blanking_period
                        if not is_in_blanking_period:
                            target = torch.tensor(self.target, dtype=torch.float32)
                            # self.loss = self.loss_func(guess, target)
                            self.loss = torch.abs(guess[0] - target[0]) + 5.0 * torch.abs(guess[1] - target[1])
                            self.optimizer.zero_grad()
                            self.loss.backward()
                            self.optimizer.step()
                            iters = iters + 1
                    break  # Only want one face
            batch_end_time = time.perf_counter()
            if iters >= self.minibatch_size:
                iters = 0
                self.gaze2screen.save(self.config.screen_gaze.save_file_name)
            logger.info(f'Loss: {self.loss * 100: .2f}\tfps: {self.minibatch_size / (batch_end_time - batch_start_time) : .2f} ')

        self.cap.release()

    def _extract_features(self, image: np.ndarray, face: Face) -> np.array:
        features = []
        self.gaze_estimator.estimate_gaze(image, face)
        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        features.append(pitch)
        features.append(yaw)
        features.append(roll)
        features.append(face.distance)

        # logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
        #             f'roll: {roll:.2f}, distance: {face.distance:.2f}')

        self.gaze_estimator.estimate_gaze(image, face)
        for key in [FacePartsName.REYE, FacePartsName.LEYE]:
            eye = getattr(face, key.name.lower())
            pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
            features.append(pitch)
            features.append(yaw)
            # logger.info(
            #     f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        return np.array(features)


def main():
    config = load_config()
    screen_gaze = ScreenGaze(config)
    screen_gaze.train()


if __name__ == '__main__':
    main()
