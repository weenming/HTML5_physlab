from torch import nn
import torch

from angular_spectrum.util.resolution import change_resolution

class DMD(nn.Module):
    def __init__(self, grid_num=None):
        '''
        grid_num: spatial discretization number in the simulation
        Note that we only specify the number of grids outside of the module,
            using grid_num = ( pixel_num * (pixel_size / grid_size) ).floor()
            This induces error.

        '''
        super().__init__()
        self.grid_num = grid_num

    def forward(self, X):
        if self.grid_num is not None:
            X = change_resolution(
                X, self.pixel, self.pixel, self.grid, self.grid
            )
        return X