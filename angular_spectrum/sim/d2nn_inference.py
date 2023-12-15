import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../../')
sys.path.append(os.path.dirname(__file__) + '/../')

from d2nn.d2nn import D2NN, DMD, SLM, CMOS
from sim.propagate_d2nn import propagate
from util.resolution_d2nn import zero_padding


import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# global params
wl = 0.5 # um, 500 nm visible light
grid_num = 500 # 
dmd_size = 8 * 500

grid_size = dmd_size / grid_num # um; total size: 1mm
load_resize = 0.4

dmd_pixel_size = None # grid_size * grid_num / size_of_image_in_pixel_num
slm_pixel_size = 8 # um

zs = [100 * 1000, 100 * 1000] # in um


def get_d2nn(show_slm=False):
    exp_idx = 0

    # load dataset
    class_num = 10 # handwritten digits

    # prepare model
    d2nn_base = D2NN(
        class_num, 
        grid_num, 
        grid_size, 
        int(dmd_size / slm_pixel_size), 
        zs, 
        frame_ratio=0.2, 
        load_resize=load_resize, 
        cmos_resize=load_resize,
        pad_ratio=1, 
    ) # cpu

    d2nn_base.load_state_dict(torch.load(os.path.dirname(__file__) + f'/../saved_model/d2nn_base_{exp_idx}', map_location=torch.device('cpu')))
    d2nn_base.eval()

    if show_slm:
        fig, ax = plt.subplots(1, 1)

        s = ax.imshow(
            show_X(nn.functional.sigmoid(d2nn_base.slm_layers[0].tau / wl), 1), # note the definition of tau. see SLM
            cmap='twilight', 
            interpolation=None
        )
        ax.set_xlabel(f'*{slm_pixel_size} um')
        ax.set_ylabel(f'*{slm_pixel_size} um')
        fig.colorbar(s)

    return d2nn_base


def before_cmos(model, X, wl, cmos=True):
    m = model
    X = zero_padding(m.dmd(X), m.pad_ratio + 1)
    # assert torch.count_nonzero((X.real != 0) & (X.real != 1)) == 0, 'bad binarization'

    if not cmos:
        return X
    X = propagate(X, m.grid_size, wl, m.zs[0])

    for i, (slm, z) in enumerate(zip(m.slm_layers, m.zs[1:])):
        X = slm(X, wl)
        X = propagate(X, m.grid_size, wl, z)
        continue

    return X


def draw_frame(ax, sq_idx=get_d2nn().cmos.get_frame_idx(torch.zeros((1000, 1000))), model=get_d2nn(), grid_size=grid_size):
    classes = model.cmos.classes
    sq_idx = (
        sq_idx[1].reshape(classes, -1), sq_idx[0].reshape(classes, -1)
    )
    for c in range(classes):
        ymax, ymin = sq_idx[1][c].max() * grid_size, sq_idx[1][c].min() * grid_size
        xmax, xmin = sq_idx[0][c].max() * grid_size, sq_idx[0][c].min() * grid_size
        ax.plot([xmin, xmin], [ymin, ymax], color='red')
        ax.plot([xmax, xmax], [ymin, ymax], color='red')
        ax.plot([xmin, xmax], [ymin, ymin], color='red')
        ax.plot([xmin, xmax], [ymax, ymax], color='red')


def show_X(X, ratio=1):
    n = X.size(-1) // 2
    return X.detach().abs().square().squeeze().cpu().numpy()[
        int(n * (1 - ratio)): int(n * (1 + ratio)), 
        int(n * (1 - ratio)): int(n * (1 + ratio))
    ]


def d2nn_inference(X_test, cmos=True, y_label=None, only_return_X=True):
    '''
    Returns the intensity on the CMOS plane.

    Arguments:
        X_test: a 28 * 28 handwritten digit as array
        cmos: if False, returns the image of the input
        y_label: Groundtruth label. Printed

    Returns:
        X:
             200 * 200 matrix. The EM wave amplitude (complex number) on 
             the CMOS plane. Only the central region is shown.
        y_pred: predicted label
    '''
    d2nn_base = get_d2nn(show_slm=False)
    d2nn_base.cpu()
    
    X_test = torch.tensor(X_test)
    X_test_cpu = X_test.cpu()
    X = before_cmos(d2nn_base, X_test_cpu, wl, cmos=cmos)

    if cmos:
        y = d2nn_base(X_test_cpu, wl)

    ratio = float(1 / (d2nn_base.pad_ratio + 0.429) * d2nn_base.dmd.resize_factor)
    X = show_X(X, ratio)

    if only_return_X:
        return X
    return X, y.argmax().item(), y

if __name__ == '__main__':
    # ugly "1"
    X = np.zeros((28, 28))
    X[:, 12:15] = 1

    X, y, logits = d2nn_inference(X, cmos=True, only_return_X=False)
    print(logits)

