from angular_spectrum.sim.layer import Layer
import torch

from angular_spectrum.sim.propagate import propagate


class D2NN(nn.Module):
    def __init__(self, grid_num, num_slm=1, slm_layer=None):
        super().__init__()

        # intialize SLM
        self.slm_layers = nn.ModuleList(
            [SLM(grid_num) for i in range(num_slm)]
        )
        if slm_layer is not None:
            self.slm_layers = slm_layers

        # initialize DMD
        self.dmd = DMD(grid_num)

    def reset_parameters(self):
        for layer in self.slm_layers:
            layer.reset_parameters()
            
