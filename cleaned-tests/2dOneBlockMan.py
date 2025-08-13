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

def reflected_power(mix_ratio, period, ang_pol, n_harm, wl):
    assert n_harm%2 == 1

    i_wl = np.where(wavelengths == wl)
    assert len(i_wl) == 1
    i_wl = i_wl[0][0]

    depth = .9
    S = S4.New(Lattice = ((period, 0), (0, period)), NumBasis = n_harm**2)
    S.SetMaterial(Name='W',   Epsilon=ff.w_n[i_wl+130]**2)
    S.SetMaterial(Name='Vac', Epsilon=1)
    S.SetMaterial(Name='AlN', Epsilon=(ff.aln_n[i_wl+130]**2-1)*mix_ratio+1)

    S.AddLayer(Name='VacuumAbove', Thickness=0.5, Material='Vac')
    S.AddLayer(Name='Grating',      Thickness=depth, Material='Vac')
    S.SetRegionRectangle(Layer = 'Grating', Material = 'AlN', Center = (period/4, period/2), Halfwidths = (period/4, period/5), Angle = 0)
    S.AddLayer(Name='VacuumBelow', Thickness=1, Material='W')
    S.SetFrequency(1.0 / wl)

    S.SetExcitationPlanewave((0,0), sAmplitude=np.cos(ang_pol*np.pi/180), pAmplitude=np.sin(ang_pol*np.pi/180), Order=0)

    return np.abs(S.GetPowerFlux('VacuumAbove', zOffset=0)[1])

emission = []
period = 1.
ang_pol = 45
n_harm = 5
wavelength = .370
mix_ratios = np.linspace(0, 1, 100)
for mix_ratio in mix_ratios:
    emission.append(reflected_power(mix_ratio, period, ang_pol, n_harm, wavelength))

emission = np.array(emission)
dx = mix_ratios[1] - mix_ratios[0]
slopes = np.gradient(emission, dx)

# (optional) save slopes to file, similar to your second script
np.save("cleaned-tests/fom_slopes.npy", slopes)

# Plot FOM (left) and gradient (right)
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(mix_ratios, emission, label="FOM(mix_ratio)")
plt.title("FOM vs mix_ratio")
plt.xlabel("mix_ratio")
plt.ylabel("Reflected power")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mix_ratios, slopes, label="dFOM/d(mix_ratio)")
plt.title("Gradient of FOM")
plt.xlabel("mix_ratio")
plt.ylabel("slope")
plt.legend()

plt.tight_layout()
plt.show()