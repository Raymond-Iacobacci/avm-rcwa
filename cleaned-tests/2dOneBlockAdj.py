import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import S4
import torch
from tqdm import tqdm

import ff

wavelengths = np.linspace(.35, 3, 2651)

def grad_array(mix_ratio, period, ang_pol, n_harm, wl, xy_density = 1, square_divisions = 1):
    assert n_harm%2 == 1

    i_wl = np.where(wavelengths == wl)
    assert len(i_wl) == 1
    i_wl = i_wl[0][0]

    def make_grid(period, n_cells, k):
        dx = period / n_cells
        fr = np.arange(1, 2 * k, 2) / (2 * k)
        st = np.arange(n_cells)[:, None]
        return ((st + fr) * dx).ravel()
    
    x_space = make_grid(period, n_cells = xy_density, k=square_divisions)