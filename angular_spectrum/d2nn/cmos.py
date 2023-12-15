from torch import nn
import torch
import numpy as np

class CMOS(nn.Module):
    def __init__(
            self, 
            classes, 
            grid_num, 
            frame_ratio=0.3, 
            load_resize=1,
            optimize_size=False
    ):
        '''
        Pretend that CMOS has infinitely good resolution.

        grid_num:
            spatial discretization grids ON THE CMOS (not counting zero padding)
            It can be used to determine the CMOS region of the zero padded X.
        classes:
            number of frames to detect the resulting light intensity
        '''
        super().__init__()

        self.grid_num = grid_num
        self.classes = classes
        self.load_resize = load_resize
        
        if not optimize_size:
            self.cols = int(np.ceil(np.sqrt(self.classes)))
            self.frame_s = int(grid_num / self.cols * frame_ratio)
        else:
            raise NotImplementedError
            self.frame_s = nn.Parameter(grid_num // (classes * 3))
        
    def reset_parameter(self):
        if self.frame_s.requires_grad:
            raise NotImplementedError('size is currently discrete.')
        return
    
    def square_frame_idx(self, n, i):
        cols = self.cols
        row, col = i // cols, i % cols

        grid_num = int(self.grid_num * self.load_resize)
        frame_s = int(self.frame_s * self.load_resize)
        
        xc = (n - grid_num) // 2 + (col + 0.5) * (grid_num // cols)
        yc = (n - grid_num) // 2 + (row + 0.5) * (grid_num // cols)
        
        x = torch.arange(
            int(xc) - frame_s // 2, int(xc) + (frame_s + 1) // 2
        )
        y = torch.arange(
            int(yc) - frame_s // 2, int(yc) + (frame_s + 1) // 2
        )
        idx = (
            y.repeat_interleave(self.frame_s), x.repeat(self.frame_s)
        )
        return idx
    
    def sym_place(self, c):
        if self.cols ** 2 == self.classes:
            return c
        if self.classes == 10:
            return ([1] + [x for x in range(4, 12)] + [14])[c]
        
        if c > self.classes // 2:
            return c + (self.cols ** 2 - self.classes)
        return c
    

    def get_frame_idx(self, X):
        idxx, idyy = torch.tensor([]).int(), torch.tensor([]).int()
        for c in range(self.classes):
            idxx_this, idyy_this = self.square_frame_idx(X.size(-1), self.sym_place(c))
            idxx = torch.hstack((idxx, idxx_this))
            idyy = torch.hstack((idyy, idyy_this))
        return idxx, idyy
    
    def forward(self, X):
        idxx, idyy = self.get_frame_idx(X)

        y = (X[..., idxx, idyy].abs().square()).reshape(
                *X.size()[:-2], self.classes, -1
        ).sum(-1)
            
        return y

