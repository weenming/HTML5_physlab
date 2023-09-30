from torch import nn

def change_resolution(X, after_x, after_y):
    '''
    before / after: number of pixels before and after the transform
    '''
    
    return nn.functional.interpolate(X, (after_x, after_y))
    

def zero_padding(X, ratio=None, Nx=None, Ny=None):
    '''
    Nx, Ny: zero padding elems on each side
    '''
    if ratio is not None:
        # overrides Nx and Ny
        Nx = (X.shape[-2] // 2 * ratio).floor
        Ny = (X.shape[-1] // 2 * ratio).floor
    
    # the padding params are like:
    # (dim-1, dim-1, dim-2, dim-2, ...)

    X = nn.functional.pad(
        X, 
        (Nx, Nx, Ny, Ny), 
        'constant', 
        0
    )
    return X