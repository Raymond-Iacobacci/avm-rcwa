# TODO: change correction k-vector for each nonzero harmonic
# TODO: figure out why adding in evanescent modes makes it distance-dependent long past where they should have decayed
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
import sys
import argparse

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
            # nn.ReLU(inplace=True),
            nn.Linear(64, n_elements)
        )

    def forward(self, z):
        # Sigmoid to ensure outputs in [0,1]
        return torch.sigmoid(self.model(z))

# --------------------------------------------------
# Physics-based gradient + full-field sampling
# --------------------------------------------------
N = 5
def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float,
                       plot_fields: bool = False):
    p = 20
    n_grating_elements = grating.shape[-1]
    x_density = 30
    n_x_pts = x_density * n_grating_elements
    n_y_pts = n_x_pts
    depth = 0.9
    vac_depth = 1.00
    z_meas = np.linspace(vac_depth, vac_depth + depth, 70)
    # measurement volume for gradient: within grating layer
    # z_meas = z_space[(z_space >= vac_depth) & (z_space <= vac_depth + depth)]
    wavelengths = torch.linspace(0.35, 3.0, 2651)
    z_buf = 0. # irrespective to this in homogeneous case but...

    def make_grid(L, n_cells, k):
        dx = L / n_cells
        fr = np.arange(1, 2 * k, 2) / (2 * k)
        st = np.arange(n_cells)[:, None]
        return ((st + fr) * dx).ravel()

    x_space = make_grid(L, n_cells=x_density, k=n_grating_elements)
    y_space = x_space.copy()

    dflux = torch.zeros((2, n_y_pts, n_x_pts))
    power = []
    for i_wl, wl in enumerate(wavelengths[p:p + 1]):
        # S = S4.New(Lattice=L, NumBasis=N)
        S = S4.New(Lattice=((L, 0), (0, L)), NumBasis=N)
        S.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl + p + 130]**2)

        S.SetMaterial(Name='Vac', Epsilon=1)
        S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wl + p+130]**2 - 1) * grating[0].item() + 1)

        S.AddLayer(Name='VacuumAbove', Thickness=vac_depth, Material='Vac')
        S.AddLayer(Name='Grating', Thickness=depth, Material='Vac')
        S.SetRegionRectangle(Layer = 'Grating', Material = 'AlN', Center = (L/2, L/2), Halfwidths = (L/4, L/2), Angle = 0)
        S.AddLayer(Name='Ab', Thickness=1.0, Material='W')
        S.SetFrequency(1.0 / wl)
        S.SetExcitationPlanewave((0, 0),
                                 sAmplitude=np.cos(ang_pol * np.pi/180),
                                 pAmplitude=np.sin(ang_pol * np.pi/180),
                                 Order=0)
        forw, back = S.GetPowerFlux('VacuumAbove', zOffset=0)
        power.append(np.abs(back))

        fwd_meas = np.zeros((z_meas.size, n_y_pts, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for iy, y in enumerate(y_space):
                for ix, x in enumerate(x_space):
                    fwd_meas[iz, iy, ix] = S.GetFields(x, y, z)[0] # [0] gets the electric field

        (forw_amp, back_amp) = S.GetAmplitudes('VacuumAbove', zOffset=z_buf) # NOTE: to make individual excitations for each polarization (which for some reason is required by the 2D tests) we must be able to get individual amplitudes for each polarization -- thus, we must split the polarization across conjugate bases.
        # Further note: this implicitly converts it into the underlying xy cartesian plot coordinates.
        # print(back_amp)
        k0 = 2 * np.pi / wl.item()
        # S-polarization = y-polarization = 0-polarization

        Ss_adj = S.Clone()
        Sp_adj = S.Clone()
        basis = Ss_adj.GetBasisSet() # Removes the repeated calls
        propagating_harmonics = [ i for i in range(2*len(basis)) if 2*np.pi*basis[i % len(basis)][0] ** 2 + 2 * np.pi*basis[i % len(basis)][1] ** 2 <= k0 ** 2] # TODO: not extending for p-polarization
        propagating_harmonics = [0, 2, 3]
        # print(f'Propagating harmonics:\n{propagating_harmonics}')
        # print(basis)
        s_excitations = []
        p_excitations = []
        # print(len(basis))
        for i, raw_amp in enumerate(back_amp):
            # print(i<len(basis))
            if i not in propagating_harmonics:
                continue
            corr_amp = complex(np.exp(-1j * k0 * z_buf) * np.conj(raw_amp))
            if i < len(basis):
                s_excitations.append((i + 1, b'y', corr_amp))
            else:
                p_excitations.append((i + 1 - len(basis), b'x', corr_amp)) # TODO: change to mod len basis
        Ss_adj.SetExcitationExterior(tuple(s_excitations))
        # Sp_adj.SetExcitationExterior(tuple(p_excitations))

        adj_meas = np.zeros((z_meas.size, n_y_pts, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for iy, y in enumerate(y_space):
                for ix, x in enumerate(x_space):
                    # adj_meas[iz, iy, ix] = np.sqrt(np.array(Ss_adj.GetFields(x, y, z)[0]) ** 2 + np.array(Sp_adj.GetFields(x, y, z)[0]) ** 2) # NOTE: this gets the electric fields inside the medium for all 3 directions
                    adj_meas[iz, iy, ix] = np.array(Ss_adj.GetFields(x, y, z)[0]) # NOTE: this gets the electric fields inside the medium for all 3 directions

        # print(np.mean(adj_meas))
        delta_eps = ff.aln_n[i_wl + p + 130] ** 2 - 1
        delta_eps_r = torch.tensor(delta_eps.real, dtype=torch.float32)
        delta_eps_i = torch.tensor(delta_eps.imag, dtype=torch.float32)
        phi = torch.einsum('ijkl,ijkl->ijk',
                           torch.as_tensor(fwd_meas),
                           torch.as_tensor(adj_meas)) # NOTE: just gets rid of the last dimension
        # print(torch.mean(phi))
        grad_r = -k0 * torch.imag(phi) * delta_eps_r
        grad_i = +k0 * torch.real(phi) * delta_eps_i
        dz = (depth) / len(z_meas)
        dflux[i_wl] = (grad_r - grad_i).sum(dim = 0)*dz * L / n_x_pts * L / n_y_pts
        # print(torch.mean(grad_r.sum(dim = 0)))
        # print(L/n_y_pts)
        # print(torch.mean(dflux))
        # print(torch.sum(dflux[0][:,7:22].real))
        # print(n_y_pts,n_x_pts)
        del S, Ss_adj, Sp_adj
    return torch.sum(dflux[0][:,7:22].real), power[0]

# --------------------------------------------------
# Main: scan & plot
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scan gradient and optionally plot full fields")
    parser.add_argument('--plot-fields', action='store_true',
                        help='Plot full-field forward and adjoint E-fields')
    args = parser.parse_args()

    L = 1.
    ang_pol = 0
    step = 0.01
    jstep = 0.01
    i_vals = np.arange(0.0, 1.0 + step, step)
    j_vals = np.arange(0.0, 1.0 + jstep, jstep)
    grad_vals = []
    P=[]
    for val in tqdm(j_vals, desc="Scanning gradient"):
        # print(val)
        g = torch.tensor([val], dtype=torch.float32)
        dflux, P1 = gradient_per_image(g, L, ang_pol, plot_fields=args.plot_fields)
        # dflux2, P12 = gradient_per_image(g, L, 90, plot_fields = args.plot_fields)
        # dflux += dflux2
        # P1 += P12
        grad_vals.append(dflux.item())
        P.append(P1)
    plt.plot(P)
    plt.show()
    correct_slopes = np.load(f'slope{N}.npy')
    plt.figure(figsize=(6, 4))
    plt.plot(j_vals, grad_vals, label = 'Calculated slopes', lw=2)
    plt.plot(i_vals, correct_slopes, label = 'Correct slopes', lw = 2)
    plt.legend()
    plt.xlabel('Grating amplitude')
    plt.ylabel('dFOM / d(amplitude)')
    plt.title('Gradient of FOM vs. Grating Amplitude')
    plt.grid(True)
    plt.show()
    # np.save(f'computed_slope{N}.npy', grad_vals)

    print(np.max(np.abs(correct_slopes[1:-1] - grad_vals[1:-1])))
    print(correct_slopes[10] , grad_vals[10])
    print(correct_slopes[50] , grad_vals[50])

if __name__ == '__main__':
    main()
