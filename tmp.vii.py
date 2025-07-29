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
N = 3
def gradient_per_image(grating: torch.Tensor, L: float, ang_pol: float,
                       plot_fields: bool = False):
    p = 20
    n_grating_elements = grating.shape[-1]
    x_density = 30
    n_x_pts = x_density * n_grating_elements
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

    dflux = torch.zeros((2, n_x_pts), dtype = torch.cfloat)
    power = []
    for i_wl, wl in enumerate(wavelengths[p:p + 1]):
        S = S4.New(Lattice=L, NumBasis=N)
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

        fwd_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                fwd_meas[iz, 0, ix] = S.GetFields(x, 0, z)[0]

        (forw_amp, back_amp) = S.GetAmplitudes('VacuumAbove', zOffset=z_buf)
        # print(back_amp)
        k0 = 2 * np.pi / wl.item()
        S_adj = S.Clone()
        basis = S_adj.GetBasisSet() # Removes the repeated calls
        pos_harmonics = [ i for i in range(len(basis)) if basis[i][0] > 0 and abs(2*np.pi*basis[i][0]/L) <= k0] # NOTE: this forces the system to work on far-fields only. Evanescent waves are discarded.
        neg_harmonics = [ i for i in range(len(basis)) if basis[i][0] <= 0 and abs(2*np.pi*basis[i][0]/L) <= k0]
        k0 = 2 * np.pi / wl.item()
        excitations = []
        for i, raw_amp in enumerate(back_amp):
            if i not in pos_harmonics and i not in neg_harmonics:
                continue
            corr_amp = complex(np.exp(-1j * k0 * z_buf) * np.conj(raw_amp))
            if i in pos_harmonics:
                excitations.append((2*basis[i][0], b'y', corr_amp))
            if i in neg_harmonics:
                excitations.append((-2*basis[i][0]+1, b'y', corr_amp))
        S_adj.SetExcitationExterior(tuple(excitations))
        # print(excitations)
        pos_adj_meas = np.zeros((z_meas.size, 1, n_x_pts, 3), complex)
        for iz, z in enumerate(z_meas):
            for ix, x in enumerate(x_space):
                pos_adj_meas[iz, 0, ix] = S_adj.GetFields(x, 0, z)[0]
        '''
        term = -k0 * torch.imag(
            torch.sqrt(
            torch.einsum('ijkl,ijkl->ijk',
                         torch.as_tensor(fwd_meas)**2,
                         torch.as_tensor(pos_adj_meas)**2
                         )
            )
        )
        '''
        # print(np.mean(pos_adj_meas))
        delta_eps = ff.aln_n[i_wl+p+130] ** 2 - 1
        delta_eps_r = torch.tensor(delta_eps.real, dtype=torch.float32)
        delta_eps_i = torch.tensor(delta_eps.imag, dtype=torch.float32)
        phi = torch.einsum('ijkl,ijkl->ijk',
                           torch.as_tensor(fwd_meas),
                            torch.as_tensor(pos_adj_meas))
        # print(torch.mean(phi))
        grad_r = -k0 * torch.imag(phi) * delta_eps_r
        grad_i = +k0 * torch.real(phi) * delta_eps_i
        dz = (depth) / len(z_meas)
        # print(torch.mean(grad_r.sum(dim = 0)))
        dflux[i_wl] = (grad_r - grad_i).sum(dim=0) * dz * L / n_x_pts
        # print(torch.mean(dflux)*L/n_x_pts)
        # print(torch.sum(dflux[0][7:22]).real)
        del S, S_adj
    return (torch.sum(dflux[0][7:22]).real), power[0] # 8, 21

# --------------------------------------------------
# Main: scan & plot
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scan gradient and optionally plot full fields")
    parser.add_argument('--plot-fields', action='store_true',
                        help='Plot full-field forward and adjoint E-fields')
    args = parser.parse_args()

    L = 1.
    ang_pol = 0.0
    step = 0.01
    i_vals = np.arange(0.0, 1.0 + step, step)
    grad_vals = []
    P=[]
    for val in tqdm(i_vals, desc="Scanning gradient"):
        # print(val)
        g = torch.tensor([val], dtype=torch.float32)
        dflux, P1 = gradient_per_image(g, L, ang_pol, plot_fields=args.plot_fields)
        grad_vals.append(dflux.item())
        P.append(P1)
    plt.plot(P)
    plt.show()
    correct_slopes = np.load(f'slope{N}.npy')
    plt.figure(figsize=(10, 5))
    plt.plot(i_vals, grad_vals, label = 'Calculated slopes', lw=2)
    plt.plot(i_vals, correct_slopes, label = 'Correct slopes', lw = 2)
    plt.legend(fontsize=15)
    plt.xlabel('Grating amplitude',fontsize=15)
    plt.ylabel('Gradient',fontsize=15)
    plt.title(r'$\frac{\partial\text{eff}}{\partial\text{comp ratio}}$ vs comp ratio @ $n_\text{harm}=$'+f'{int((N+1)/2)}',fontsize=20)
    plt.grid(True)
    plt.show()
    # np.save(f'computed_slope{N}.npy', grad_vals)
    # np.save('computed_slope5.npy', grad_vals)

    print(np.max(np.abs(correct_slopes[1:-1] - grad_vals[1:-1])))
    print(correct_slopes[10] , grad_vals[10])
    print(correct_slopes[50] , grad_vals[50])

if __name__ == '__main__':
    main()


'''
        if plot_fields:
            z_space = np.linspace(0, 1+vac_depth + depth, 30)
            fwd_vol = np.zeros((z_space.size, n_x_pts), complex)
            adj_vol = np.zeros((z_space.size, n_x_pts), complex)
            for iz, z in enumerate(z_space):
                for ix, x in enumerate(x_space):
                    fwd_vol[iz, ix] = S.GetFields(x, 0, z)[0][1]
                    adj_vol[iz, ix] = S_adj.GetFields(x, 0, z)[0][1] + S_adj.GetFields(-x, 0, z)[0][1]


            # compute product of forward and adjoint fields
            prod_vol = fwd_vol * (adj_vol)

            fig, axs = plt.subplots(3, 2, figsize=(10, 12))
            titles = [
                f'Forward Real E-field {grating[0]}', f'Forward Imag E-field {vac_depth}',
                'Adjoint Real E-field', 'Adjoint Imag E-field',
                'Product Real (E·E_adj)', 'Product Imag (E·E_adj)'
            ]
            data_list = [
                np.real(fwd_vol), np.imag(fwd_vol),
                np.real(adj_vol), np.imag(adj_vol),
                np.real(prod_vol), np.imag(prod_vol)
            ]

            for ax, title, data in zip(axs.ravel(), titles, data_list):
                im = ax.imshow(
                    data,
                    extent=[x_space.min(), x_space.max(), z_space.max(), z_space.min()],
                    aspect='auto'
                )
                ax.set_title(f'{title} {grating[0]}' if 'Forward' in title else title)
                ax.set_xlabel('x (um)')
                if 'Real' in title:
                    ax.set_ylabel('z (um)')
                # layer boundaries
                ax.axhline(vac_depth, color='white', linestyle='--', linewidth=1)
                ax.axhline(vac_depth + depth, color='white', linestyle='--', linewidth=1)
                fig.colorbar(im, ax=ax)

            plt.tight_layout()
            plt.show()

'''