from torch import nn
import torch

from angular_spectrum.util.resolution_d2nn import change_resolution, zero_padding, zero_padding_from_to

class SLM(nn.Module):
    def __init__(self, pixel_num, grid_num=None, tau=None, **kwargs):
        '''
        
        grid_num: spatial discretization number OF THE SLM in the simulation
            (not counting the zero padding.)
        Note that we only specify the number of grids outside of the module,
            using grid_num = ( pixel_num * (pixel_size / grid_size) ).floor()
            This induces error.

        tau: optical thickness of the SLM
        '''
        super().__init__()
        self.grid_num = grid_num
        self.pixel_num = pixel_num

        # max allowed phase modulation
        # NOTE: idk if we need to use 0-pi or 0-2pi
        self.max_ot = 500
        # not including dispersion for now. lambda will interfere with pickle
        # self.dispersion_factor = lambda wl: 1

        # intialize phase modulation parameter
        self.tau = nn.Parameter(torch.zeros((self.pixel_num, self.pixel_num), dtype=torch.float64))
        if tau is None:
            self.reset_parameters()
        else:
            assert tau.size() == self.tau.size(), 'bad initializer'
            self.tau = tau

        # check parameters
        if grid_num is not None:
            assert self.pixel_num <= self.grid_num, 'grid should not be smaller than pixel'
        else:
            print('warning: grid num not set. using pixel as grid. \
                  may cause dismatching X size')

    def reset_parameters(self):
        nn.init.xavier_normal_(self.tau) * self.max_ot
        return
    
    def _tau_on_grid(self, tau):
        return change_resolution(
            tau, self.grid_num, self.grid_num
        )

    def forward(self, X, wl):
        # NOTE: I am not sure if I should zero pad the phase (free space 
        # outside of SLM) or zero pad the whole modulator on X (full aborption
        # outside of SLM)
        tau = self._tau_on_grid(nn.functional.sigmoid(self.tau / wl)) - 0.5
        tau = zero_padding_from_to(tau, X)
        
        X = X * torch.exp(
            2 * 1j * torch.pi * tau # ............I forgot 1j.........
        )
        return X