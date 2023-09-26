import torch
import torch.fft as fft
from util.const import n0

def propagate(X, pixel, wl, z, n=lambda wl: n0):
    N = X.shape[0]
    assert N == X.shape[1], 'support only square input for now'
    
    Xk = fft.fft2(X)
    
    kx = fft.fftfreq(N, d=pixel)
    ky = fft.fftfreq(N, d=pixel)
    k0 = n(wl) * 2 * torch.pi / wl

    kz = torch.sqrt(k0 ** 2 - (kx ** 2 + ky.unsqueeze(-1) ** 2))

    Xk = torch.exp(1j * kz * z) * Xk # favorably this operator is elem-wise
    X = fft.ifft2(Xk)
    return X