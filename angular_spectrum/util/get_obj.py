import torch
import numpy as np
import matplotlib.pyplot as plt

def double_slit(halfn):
    w = 10
    d = 2
    x_left_idx = torch.arange(halfn - w // 2 - d // 2, halfn - w // 2 + d // 2)
    x_right_idx = torch.arange(halfn + w // 2 - d // 2, halfn + w // 2 + d // 2)
    x_left_idx = x_left_idx.repeat(halfn * 2)
    x_right_idx = x_right_idx.repeat(halfn * 2)
    y_idx = torch.arange(0, halfn * 2).repeat_interleave(d)
      
    
    X = torch.zeros((halfn * 2, halfn * 2))
    X[x_left_idx, y_idx] = 1
    X[x_right_idx, y_idx] = 1
    return X

def obj(get_x, n):
    halfn = n // 2
    X = get_x(halfn)

    X = X / X.max() # normalize
    X = X.to(torch.complex128) # NOTE: complex 64 induces accumulated error which is catastrophic
    return X

def circle(halfn):
    r = 10
    X = torch.arange(-halfn, halfn) ** 2 + torch.arange(-halfn, halfn).unsqueeze(-1) ** 2
    idx = X < r ** 2
    n_idx = X >= r ** 2

    X[n_idx] = 0
    X[idx] = 1
    return X

def square(halfn):
    s = 10
    idx = torch.arange(halfn - s, halfn + s)
    idx = (idx.repeat(2 * s), 
        idx.repeat_interleave(2 * s))
    

    X = torch.zeros((halfn * 2, halfn * 2))
    X[idx] = 1
    return X

def cut(X, show_ratio):
    halfn = X.shape[0] // 2
    lower = int(halfn * (1 - show_ratio))
    higher = int(halfn * (1 + show_ratio))
    
    return X[lower: higher, lower:higher]

def show_mat(X, pixel, show_ratio=1):
    X = cut(X, show_ratio)
    halfn = X.shape[0] // 2
    
    X = X.abs().cpu().numpy()

    fig, ax = plt.subplots(1, 1)
    s = ax.imshow(
        X, 
        extent=[-halfn * pixel, halfn * pixel, -halfn * pixel, halfn * pixel]
    )
    ax.set_xlabel('$\mu$m')
    ax.set_ylabel('$\mu$m')
    ax.set_title('objective')
    fig.colorbar(s)

    return