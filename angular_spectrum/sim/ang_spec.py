from angular_spectrum.sim.layer import Layer
import torch

from angular_spectrum.sim.propagate import propagate


def diffract(X, pixel, z, wl, layers: list(Layer)=[]):
    '''
    o: object
    i: image
    z: distance of last layer from image. inter-layer distance is encapsulated 
        in `layers`
    '''

    for layer in layers:
        layer.phase_modulate(X, pixel, wl)
        X = propagate(X, pixel, z, wl)
    
    X = propagate(X, pixel, z, wl)
    return X




