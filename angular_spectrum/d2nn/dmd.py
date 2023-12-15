from torch import nn
import torch

from angular_spectrum.util.resolution_d2nn import change_resolution, zero_padding

class DMD(nn.Module):
    def __init__(self, grid_num=None, resize_factor=1):
        '''
        grid_num: spatial discretization number in the simulation
        Note that we only specify the number of grids outside of the module,
            using grid_num = ( pixel_num * (pixel_size / grid_size) ).floor()
            This induces error.
        
        There are two approaches to set the input picture:
        (1) resize the input (28 by 28 MINIST images) to 1920 by 1080 DMD screen
        and then resize that to grid size inside this module, or
        (2) directly resize the 28 by 28 image to grid size in this module
        We adopt the second approach for now. It should not make a big 
        difference though.

        Note that the resize factor indicates the ratio of how much space X 
        takes up on the DMD screen, e.g. for resize_factor=0.1, the X takes up
        10% size of the DMD. The ratio between the pixel size of input image
        and the grid size is then 
        grid_size * grid_num * resize_factor / X.size(-1).

        '''
        super().__init__()
        self.grid_num = grid_num
        self.resize_factor = resize_factor

            
    def load_vector(self, v):
        # note that v comes in batch * cols
        # i.e. v.size = batch * column * feature
        # batch * column are essentially independent but cols concat into mat
        

        return v

    def forward(self, X):
        X = zero_padding(X, 1 / self.resize_factor)

        if self.grid_num is not None:
            X = change_resolution(
                X, self.grid_num, self.grid_num
            )
        # binarize
        X[X >= 0.5] = 1
        X[X < 0.5] = 0

        X = X.to(torch.complex128)

        return X


