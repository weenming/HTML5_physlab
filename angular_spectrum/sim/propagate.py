import torch
import torch.fft as fft
from util.const import n0

def propagate(X, pixel, wl, z, n=lambda wl: n0):
    N = X.shape[0]
    assert N == X.shape[1], 'support only square input for now'
    
    Xk = fft.fft2(X)
    
    kx = fft.fftfreq(N, d=pixel)
    ky = fft.fftfreq(N, d=pixel)
    kx, ky = kx.to(dtype=X.dtype), ky.to(dtype=X.dtype)
    kx, ky = kx.to(device=X.device), ky.to(device=X.device)

    k0 = n(wl) * 2 * torch.pi / wl

    kz = torch.sqrt(k0 ** 2 - (torch.square(kx) + torch.square(ky).unsqueeze(-1)))

    Xk = torch.exp(1j * kz * z) * Xk # favorably this operator is elem-wise
    X = fft.ifft2(Xk)
    return X