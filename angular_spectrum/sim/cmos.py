from torch import nn
import torch

class CMOS(nn.Module):
    def __init__(self, classes, grid_num, optimize_size=False):
        '''
        Pretend that CMOS has infinitely good resolution.

        grid_num:
            spatial discretization grids ON THE CMOS (not counting zero padding)
            It can be used to determine the CMOS region of the zero padded X.
        classes:
            number of frames to detect the resulting light intensity
        '''

        self.grid_num = grid_num
        self.classes = classes
        self.frame_s = nn.Parameter(grid_num // (classes * 3))
        self.frame_s.requires_grad_(optimize_size)
        
    def reset_parameter(self):
        if self.frame_s.requires_grad:
            raise NotImplementedError('size is currently discrete.')
        return
    
    def square_frame_idx(self, n, i):
        cols = torch.sqrt(self.classes).floor() + 1
        row, col = i // cols, i % cols
        xc, yc = (col + 0.5) * (n // cols), (row + 0.5) * (n // cols)
        
        x = torch.arange(
            xc.int() - self.frame_s // 2, xc.int() + (self.frame_s + 1) // 2
        )
        y = torch.arange(
            yc.int() - self.frame_s // 2, yc.int() + (self.frame_s + 1) // 2
        )
        idx = (
            x.repeat(self.frame_s), y.repeat_interleave(self.frame_s)
        )
        return idx

    def forward(self, X):
        n = X.size(-1)
        y = [X[self.square_frame_idx(n, c)].abs().sum() for c in self.classes]
        return y

