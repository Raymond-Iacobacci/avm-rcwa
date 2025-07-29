import os
import random
from pathlib import Path

import ff
import matplotlib.pyplot as plt
import numpy as np
import S4
import torch
import torch.nn as nn
from tqdm import tqdm

# --------------------------------------------------
# Neural network generator: maps latent vector to grating pattern
# --------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, n_elements):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Softmax(dim=-1),
            # nn.ReLU(inplace=True),
            nn.Linear(64, n_elements)
        )

    def forward(self, z):
        # Sigmoid to ensure outputs in [0,1]
        return torch.sigmoid(self.model(z))

# --------------------------------------------------
# Physics-based gradient: computes dfom_deps and FOM diff
# --------------------------------------------------
n=3**2
def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float, i):
    p = 1480
    p = 20
    n_grating_elements = grating.shape[-1]
    x_density = 5
    n_x_pts = x_density * n_grating_elements
    depth = 0.9
    z_buff = 0.15
    z_space = np.linspace(0, 0.5 + depth + 1, 10)
    z_meas = z_space[(z_space >= 0.5) & (z_space <= 0.5 + depth)]
    wavelengths = torch.linspace(0.35, 3.0, 2651)

    def make_grid(L, n_cells, k):
        dx = L / n_cells
        fr = np.arange(1, 2*k, 2) / (2*k)
        st = np.arange(n_cells)[:, None]
        return ((st + fr) * dx).ravel()
    x_meas = make_grid(L, n_cells=x_density, k=n_grating_elements)

    dflux = torch.zeros((2, n_x_pts))
    power = []

    for i_wl, wl in enumerate(wavelengths[p:p+1]):
        # S = S4.New(Lattice=L, NumBasis=n)
        S = S4.New(Lattice=((L, 0), (0, L)), NumBasis=n)
        S.SetMaterial(Name='W',   Epsilon=(ff.w_n[i_wl+p+130]**2-1)+1)
        S.SetMaterial(Name='Vac', Epsilon=1)
        # S.SetMaterial(Name='AlN', Epsilon=(ff.cao_n[i_wl+p]**2-1)*i+1)
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wl+p+130]**2-1)*i+1)


        S.AddLayer(Name='VacuumAbove', Thickness=0.5, Material='Vac')
        S.AddLayer(Name='Grating',      Thickness=depth, Material='Vac')
        S.SetRegionRectangle(Layer = 'Grating', Material = 'AlN', Center = (L/2, L/2), Halfwidths = (L/4, L/2), Angle = 0)
        S.AddLayer(Name='VacuumBelow', Thickness=1, Material='W')
        # S.AddLayer(Name='Ab', Thickness=1.0, Material='W')
        S.SetFrequency(1.0 / wl)

        S_adj = S.Clone()
        S.SetExcitationPlanewave((0,0), sAmplitude=np.cos(ang_pol*np.pi/180), pAmplitude=np.sin(ang_pol*np.pi/180), Order=0)
        # excitations = []
        # excitations.append((2, b'y', 1))
        # S.SetExcitationExterior(tuple(excitations))
        (_, back) = S.GetPowerFlux('VacuumAbove', zOffset=0)
        power.append(np.abs(back))

        
    # print(power[0])
    return power[0]

# --------------------------------------------------
# Main optimization: train generator network & live-plot diffs
# --------------------------------------------------
# def main():
#     x=[]
#     for i in np.arange(0, 1, step = 0.001):
#         x.append(gradient_per_image(torch.zeros(size = (20,)), 1.1, 0, i))
#     print(np.argmax(np.array(x[1:])))
#     plt.plot(x)
#     plt.show()

# if __name__ == '__main__':
#     main()

def main():
    # 1. generate your FOM curve exactly as before:
    step = 0.01
    x_vals = []
    for i in np.arange(0, 1+step, step):
        x_vals.append(gradient_per_image(torch.zeros(20), L=1., ang_pol=90, i=i))
    x = np.array(x_vals)  # shape (1000,)

    # 2. compute the slope at each point:
    #    numpy.gradient does central differences in the interior,
    #    and first‐order differences at the boundaries.
    slopes = np.gradient(x, step)

    # 3. (optional) save them out:
    np.savetxt("fom_slopes.txt", slopes, header="dFOM/di")

    # 4. plot both the original curve and its slope:
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(x, label="FOM(i)")
    plt.title("FOM vs i")
    plt.xlabel("step index")
    plt.ylabel("FOM")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(slopes, label="dFOM/di")
    plt.title("Gradient of FOM")
    plt.xlabel("step index")
    plt.ylabel("slope")
    plt.legend()
    np.save(f'slope{n}.npy', slopes)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()