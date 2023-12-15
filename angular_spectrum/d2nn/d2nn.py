from angular_spectrum.d2nn.cmos import CMOS
from angular_spectrum.d2nn.dmd import DMD
from angular_spectrum.d2nn.slm import SLM
import torch
from torch import nn


from angular_spectrum.sim.propagate_d2nn import propagate
from angular_spectrum.util.resolution_d2nn import zero_padding, zero_padding_from_to

 
class D2NN(nn.Module):
    def __init__(
            self, 
            classes, 
            grid_num, 
            grid_size, # in um
            slm_pixel_num,
            zs,
            load_resize=1, 
            cmos_resize=1,
            frame_ratio=0.3,
            pad_ratio=1, 
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
        self.dmd = DMD(grid_num, load_resize)
        
        # initialize CMOS
        self.cmos = CMOS(
            classes, grid_num, 
            frame_ratio=frame_ratio, 
            load_resize=cmos_resize, 
            optimize_size=False
        )


        # check parameters
        to_complex128 = lambda x: torch.tensor(x).to(torch.complex128)
        self.zs = to_complex128(zs)
        assert len(zs) == num_slm + 1
        self.grid_size = grid_size
        self.pad_ratio = pad_ratio
        self.grid_num = grid_num
        self.slm_pixel_num = slm_pixel_num
        assert grid_num % 2 == slm_pixel_num % 2 == 0

    def reset_parameters(self):
        for layer in self.slm_layers:
            layer.reset_parameters()
        self.cmos.reset_parametrs()
    
    def forward(self, X, wl):

        X = zero_padding(self.dmd(X), self.pad_ratio)
        
        X = propagate(X, self.grid_size, wl, self.zs[0])

        for i, (slm, z) in enumerate(zip(self.slm_layers, self.zs[1:])):
            X = slm(X, wl)
            X = propagate(X, self.grid_size, wl, z)
        
        y = self.cmos(X)
        
        return y
