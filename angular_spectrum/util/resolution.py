from torch import nn

def change_resolution(X, after_x, after_y):
    '''
    before / after: number of pixels before and after the transform
    '''
    
    return nn.functional.interpolate(X, (after_x, after_y))
    

def zero_padding(X, ratio=None, Nx=None, Ny=None):
    '''
    ratio: e.g. ratio=10 -> +4.5 N each side
    Nx, Ny: zero padding elems on each side
    '''
    if ratio is not None:
        # overrides Nx and Ny
        Nx = (X.shape[-2] // 2 * (ratio - 1)).floor()
        Ny = (X.shape[-1] // 2 * (ratio - 1)).floor()
    
    # the padding params are like:
    # (dim-1, dim-1, dim-2, dim-2, ...)

    X = nn.functional.pad(
        X, 
        (Nx, Nx, Ny, Ny), 
        'constant', 
        0
    )
    return X

def zero_padding_from_to(X, Y):
    '''
    zero padding from X size to Y.
    ''' 
    assert X.size(-1) < Y.size(-1) and X.size(-2) < Y.size(-2)

    grid_dif = Y.size(-1) - X.size(-1)
    if grid_dif % 2 != 0:
        print('warning: asymmetric padding.')
    X = zero_padding(
        X, Nx=grid_dif // 2, Ny=(grid_dif + 1) // 2
    )
    return X    