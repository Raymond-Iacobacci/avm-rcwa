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
n_harm = 13

def rect_mask(period, xy_points, center, halfwidths, angle=0.0, wrap=False):
    """
    Return a boolean (ny, nx) mask over midpoint-sampled grid.
    center, halfwidths, and angle are in the same units as the period.
    angle is in radians (CCW).
    """
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(halfwidths[0]), float(halfwidths[1])

    # Midpoint coordinates
    dx = period / xy_points
    x = (np.arange(xy_points) + 0.5) * dx      # (nx,)
    y = (np.arange(xy_points) + 0.5) * dx      # (ny,)

    # Broadcasted offsets: Y first axis, X second axis  -> (ny, nx)
    Xc = x[None, :] - cx
    Yc = y[:,  None] - cy

    if wrap:
        # Periodic shortest-distance wrap (useful if the rectangle crosses cell edges)
        Xc = (Xc + period/2) % period - period/2
        Yc = (Yc + period/2) % period - period/2

    if angle != 0.0:
        ca, sa = np.cos(-angle), np.sin(-angle)
        Xr = ca * Xc - sa * Yc
        Yr = sa * Xc + ca * Yc
    else:
        Xr, Yr = Xc, Yc

    # Tiny epsilon so midpoints exactly on the edge don’t flicker due to FP
    eps = 1e-12
    return (np.abs(Xr) <= hx + eps) & (np.abs(Yr) <= hy + eps)

def sum_inside_rect(grad_map_2d, period, xy_points, center, halfwidths, angle=0.0):
    """
    Integrate (sum) a 2D gradient density over the rectangle using midpoint areas.
    grad_map_2d shape: (ny, nx) — e.g., after summing over z and taking real part.
    """
    mask = rect_mask(period, xy_points, center, halfwidths, angle)
    cell_area = (period / xy_points) ** 2
    g = torch.as_tensor(grad_map_2d).real
    return (g[mask] * cell_area).sum()

def grad_array(mix_ratio, period, ang_pol, n_harm, wl, xy_points = 1, z_points = 1, tiles_per_side=2):
    """
    If mix_ratio is scalar, uses a uniform mix everywhere the AlN features exist.
    If mix_ratio is 2x2-like (list/array), assigns a distinct mix to each of the 4 squares.
    Returns a tensor of 4 gradient sums (one per square, row-major: y then x).
    """
    assert n_harm%2 == 1
    n=int(tiles_per_side)

    i_wl = np.where(wavelengths == wl)
    assert len(i_wl) == 1
    i_wl = i_wl[0][0]

    depth=.9
    vac_depth = 1.0

    z_meas = np.linspace(vac_depth, vac_depth + depth, z_points)

    s = S4.New(Lattice = ((period, 0), (0, period)), NumBasis = n_harm**2)
    s.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl+130]**2)
    s.SetMaterial(Name='Vac', Epsilon=1)

    # normalize mix_grid into (n,n) and clamp to [0,1]
    mix_grid = np.asarray(mix_ratio, dtype=float)
    if mix_grid.ndim == 0:
        mix_grid = np.full((n, n), float(mix_grid))
    elif mix_grid.ndim == 1:
        assert mix_grid.size == n*n, f"Expected {n*n} mix values, got {mix_grid.size}"
        mix_grid = mix_grid.reshape(n, n)  # row-major: [g00, g01, ...] -> [[row0...],[row1...],...]
    else:
        mix_grid = mix_grid.reshape(n, n)
    np.clip(mix_grid, 0.0, 1.0, out=mix_grid)

    # materials per square: eps = 1 + mix * (eps_AlN - 1)
    aln_eps = ff.aln_n[i_wl+130]**2
    delta_eps = aln_eps - 1
    for iy in range(n):
        for ix in range(n):
            eps_here = 1 + mix_grid[iy, ix] * delta_eps
            s.SetMaterial(Name=f'AlN_{iy}_{ix}', Epsilon=eps_here)

    s.AddLayer(Name='VacuumAbove', Thickness=vac_depth, Material='Vac')
    s.AddLayer(Name='Grating',      Thickness=depth, Material='Vac')

    # --- geometry: n×n tiling over the unit cell ---
    # centers at (ix+0.5)/n * period, (iy+0.5)/n * period; halfwidths = period/(2n)
    hw = period / (2*n)
    halfwidths = (hw, hw)
    centers = []
    for iy in range(n):
        cy = (iy + 0.5) * (period / n)
        for ix in range(n):
            cx = (ix + 0.5) * (period / n)
            centers.append(((cx, cy), iy, ix))
            s.SetRegionRectangle(Layer='Grating', Material=f'AlN_{iy}_{ix}',
                                 Center=(cx, cy), Halfwidths=halfwidths, Angle=0.0)

    # Precompute delta eps (per 1.0 increment in mix)
    aln_eps = ff.aln_n[i_wl+130]**2
    delta_eps = aln_eps - 1

    # Homogeneous W "substrate" (thick enough region beneath)
    s.AddLayer(Name='VacuumBelow', Thickness=1, Material='W')

    s.SetFrequency(1.0 / wl)
    ss_adj = s.Clone()
    sp_adj = s.Clone()
    basis = s.GetBasisSet()
    k0 = 2 * np.pi / wl

    s.SetExcitationPlanewave((0,0), sAmplitude=np.cos(ang_pol*np.pi/180), pAmplitude=np.sin(ang_pol*np.pi/180), Order = 0)

    fwd_meas = np.zeros((z_meas.size, xy_points, xy_points, 3), complex)

    for iz, z in enumerate(z_meas):
        e2, _ = s.GetFieldsOnGrid(
            z=z,
            NumSamples=(2*xy_points, 2*xy_points),
            Format='Array'
        )  # returns shape (2*nx, 2*ny, 3)
        e2 = np.asarray(e2)
        # pick midpoints: (2i+1, 2j+1) -> (i+0.5)/n, (j+0.5)/n
        e_mid = e2[1::2, 1::2, :]                   # shape (nx, ny, 3)
        fwd_meas[iz] = e_mid
    
    fwd_back_amp = s.GetAmplitudes('VacuumAbove', zOffset=0)[1]
    r = np.abs(s.GetPowerFlux('VacuumAbove', zOffset=0)[1])
    propagating_harmonics = [ i for i in range(2*len(basis)) if (2*np.pi*basis[i % len(basis)][0]/period) ** 2 + (2 * np.pi*basis[i % len(basis)][1]/period) ** 2 <= k0 ** 2] # TODO: not extending for p-polarization

    s_excitations = []
    p_excitations = []

    for i, raw_amp in enumerate(fwd_back_amp):
        if i not in propagating_harmonics:
            continue
        corr_amp = complex(np.exp(-1j * k0*0) * np.conj(raw_amp)) # NOTE: OG z_buf is default 0 everywhere, as is what the tests do
        if i < len(basis):
            s_excitations.append((i + 1, b'y', corr_amp))
        else:
            p_excitations.append((i + 1 - len(basis), b'x', -corr_amp)) # TODO: change to mod len basis
    ss_adj.SetExcitationExterior(tuple(s_excitations))
    sp_adj.SetExcitationExterior(tuple(p_excitations))

    s_adj_meas = np.zeros((z_meas.size, xy_points, xy_points, 3), complex)
    p_adj_meas = s_adj_meas.copy()

    for iz, z in enumerate(z_meas):
        e2, _ = ss_adj.GetFieldsOnGrid(
            z=z,
            NumSamples=(2*xy_points, 2*xy_points),
            Format='Array'
        )  # returns shape (2*nx, 2*ny, 3)
        e2 = np.asarray(e2)
        e_mid = e2[1::2, 1::2, :]
        s_adj_meas[iz] = e_mid

        e2, _ = sp_adj.GetFieldsOnGrid(
            z=z,
            NumSamples=(2*xy_points, 2*xy_points),
            Format='Array'
        )  # returns shape (2*nx, 2*ny, 3)
        e2 = np.asarray(e2)
        e_mid = e2[1::2, 1::2, :]
        p_adj_meas[iz] = e_mid

    # NOTE: For the gradient *with respect to each square's mixing coefficient*,
    # the local density uses delta_eps from AlN relative to vacuum.
    delta_eps_r = torch.tensor(delta_eps.real, dtype=torch.float32)
    delta_eps_i = torch.tensor(delta_eps.imag, dtype=torch.float32)

    dz = depth / z_meas.size

    s_phi = torch.einsum('ijkl,ijkl->ijk',
                        torch.as_tensor(fwd_meas),
                        torch.as_tensor(s_adj_meas)) # summed over field components
    s_grad_r = -k0 * torch.imag(s_phi) * delta_eps_r
    s_grad_i = +k0 * torch.real(s_phi) * delta_eps_i

    p_phi = torch.einsum('ijkl,ijkl->ijk',
                        torch.as_tensor(fwd_meas),
                        torch.as_tensor(p_adj_meas))
    p_grad_r = -k0 * torch.imag(p_phi) * delta_eps_r
    p_grad_i = +k0 * torch.real(p_phi) * delta_eps_i

    # 2D gradient density wrt "local" mix (per unit area)
    grad_map_2d = (s_grad_r - s_grad_i + p_grad_r - p_grad_i).sum(dim=0) * dz  # (ny, nx)

    # integrate per square (row-major y,x) → (n,n) tensor
    per_square = torch.zeros((n, n), dtype=torch.float32)
    for (center, iy, ix) in centers:
        g_sum = sum_inside_rect(
            grad_map_2d=grad_map_2d, period=period, xy_points=xy_points,
            center=center, halfwidths=halfwidths, angle=0.0
        )
        per_square[iy, ix] = g_sum

    # Return as tensor [g00, g01, g10, g11] in row-major order (y then x)
    return r, per_square

# ------------------------ Run once: random 2x2 grating and print gradients ------------------------
period = 1.
ang_pol = 45
wl = .37

# Random mixing coefficients in [0,1] for 2x2 squares (row-major: y then x)
rng = np.random.default_rng()
n=5
mix_2x2 = rng.random((n,n))
mix_2x2 = np.array(mix_2x2)

for epoch in range(100):
    print(f'Starting epoch {epoch+1}')
    ref_pwr, grads = grad_array(mix_2x2, period, ang_pol, n_harm, wl, xy_points=30, z_points=40, tiles_per_side=n)
    print(f"Mixing coefficients ({n}x{n}, rows are y, cols are x):")
    print(mix_2x2)
    print("Per-square gradients (row-major [g00, g01, g10, g11]):")
    print(grads.numpy())
    print("Power:")
    print(ref_pwr)
    mix_2x2 = mix_2x2 + grads.numpy().reshape(n,n)*.01 - .001 * (1.0 - 2.0 * mix_2x2)
    np.clip(mix_2x2, 0.0, 1.0, out=mix_2x2)
