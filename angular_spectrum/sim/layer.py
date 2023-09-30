from torch import nn
import torch

class SLM(nn.Module):
    def __init__(self, pixel, grid, X=None, **kwargs):
        '''
        pixel: resolution of the SLM
        grid: spatial discretization size in the simulation
        '''
        self.pixel = pixel
        self.grid = grid

        self.X = nn.Parameter(torch.zeros((pixel, pixel)))
        if X is None:
            self.reset_parameters()
        else:
            assert X.size() == self.X.size(), 'bad initializer'
            self.X = X
        assert self.pixel > self.grid

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.X) * 2 * torch.pi
    
    def _X_on_grid(self, X):
        change_resolution(X, self.pixel, self.grid)

    def phase_modulate(self, pixel_x, wl):
        assert (pixel_x - self.pixel).abs() < 1e-10, 'not only support same \
            resolution between image and layer.'
        raise NotImplementedError