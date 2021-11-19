import logging

import torch
from torch import nn
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Gaze2Screen(nn.Module):
    def __init__(self, filename=None):
        """
        input features:
        head:pitch
        head:yaw
        head:roll
        head:dist
        reye:pitch
        reye:yaw
        leye:pitch
        leye:yaw
        """
        n_features = 8
        n_internal_features = 8
        n_internal_layers = 3
        super().__init__()
        self.input_layer = nn.Linear(n_features, n_internal_features)
        self.output_layer = nn.Linear(n_internal_features, 2)
        self.internal_layers = nn.Sequential()
        for layer in range(n_internal_layers):
            self.internal_layers.add_module(f'fc{layer}', nn.Linear(n_internal_features, n_internal_features))

        if filename is not None:
            try:
                self.load_state_dict(torch.load(filename))
            except FileNotFoundError:
                logger.warning(f'{filename} not found, using random weights')
            except RuntimeError:
                logger.warning(f'{filename} model is incompatible with this version')

    def forward(self, x):
        x = self.input_layer(x)
        x = self.internal_layers(x)
        x = self.output_layer(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename)
