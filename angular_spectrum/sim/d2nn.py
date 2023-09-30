from angular_spectrum.sim.cmos import CMOS
from angular_spectrum.sim.dmd import DMD
from angular_spectrum.sim.slm import SLM
import torch
from torch import nn

from angular_spectrum.sim.propagate import propagate
from angular_spectrum.util.resolution import zero_padding, zero_padding_from_to

 
class D2NN(nn.Module):
    def __init__(
            self, 
            classes, 
            grid_num, 
            grid_size, # in um
            pad_num, 
            slm_pixel_num,
            zs: list(float),
            num_slm=1, 
            slm_layers=None, # intialized SLMs
    ):
        '''
        The sizes of pixels of DMD and SLM are implicitly decided by 
        grid_size * grid_num / size_of_input_X
        and 
        grid_size * grid_num / slm_pixel_num
        NOTE: the size of DMD and SLM is assumed to be the same.
        
        pad_num:
            added padding on each side
            total size = (pad_num + grid_num + pad_num)
        '''
        super().__init__()

        # intialize SLM
        self.slm_layers = nn.ModuleList(
            [SLM(slm_pixel_num, grid_num) for i in range(num_slm)]
        )
        if slm_layers is not None:
            self.slm_layers = slm_layers

        # initialize DMD
        self.dmd = DMD(grid_num)
        
        # initialize CMOS
        self.cmos = CMOS(classes, grid_num, optimize_size=False)


        # check parameters
        self.zs = zs
        assert len(zs) == num_slm + 1
        self.grid_size = grid_size
        self.pad_num = pad_num
        self.grid_num = grid_num
        self.slm_pixel_num = slm_pixel_num
        assert pad_num % 2 == grid_num % 2 == slm_pixel_num % 2 == 0

    def reset_parameters(self):
        for layer in self.slm_layers:
            layer.reset_parameters()
        self.cmos.reset_parametrs()
    
    def forward(self, X, wl):
        
        X = zero_padding(self.dmd(X), self.pad_num)
        
        X = propagate(X, self.grid_size, wl, self.zs[0])

        for i, (slm, z) in enumerate(zip(self.slm_layers, self.zs)):
            X = slm(X, wl)
            X = propagate(X, self.grid_size, wl, z)
        
        return self.cmos(X)
