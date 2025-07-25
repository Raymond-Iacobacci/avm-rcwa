import ast
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import numpy.typing as npt
import pandas as pd  # type: ignore
import torch


# n_all = np.load('../data/n_allHTMats.npz')
# k_all = np.load('../data/k_allHTMats.npz')

# w_n = n_all['arr_0'][:, -1] + k_all['arr_0'][:, -1] * 1j
# aln_n = n_all['arr_0'][:, 17] + k_all['arr_0'][:, 17] * 1j


dfn = pd.read_excel('../data/n_AlN+W.xlsx')
dfk = pd.read_excel('../data/k_AlN+W.xlsx')

aln_n = (dfn['AlN']+1j*dfk['AlN']).to_numpy()
w_n = (dfn['W']+1j*dfk['W']).to_numpy()

n_short = np.load('../data/n_allHTMats.npz')
k_short = np.load('../data/k_allHTMats.npz')

cao_n = n_short['arr_0'][:, 4] + k_short['arr_0'][:, 4]
mgo_n = n_short['arr_0'][:, 5] + k_short['arr_0'][:, 5]

hfb2_n = n_short['arr_0'][:, 36] + k_short['arr_0'][:, 36]

zro2_n = n_short['arr_0'][:, 16] + k_short['arr_0'][:, 16]
zrb2_n = n_short['arr_0'][:, 39] + k_short['arr_0'][:, 39]

def IQE(wavelength, e_g):
    lambda_g = np.ceil(1240 / e_g) / 1000.0

    if (lambda_g > wavelength[-1]):
        l_index = wavelength[-1]
    else:
        l_index = torch.where(wavelength >= lambda_g)[0][0]
    IQE = torch.ones(len(wavelength))
    for i in range(l_index, len(wavelength)):
        IQE[i] = 0
    return IQE


def JV(em, IQE, lambda_i):
    nb_B_e = nb_B(lambda_i, T_e)
    nb_B_PV = nb_B(lambda_i, T_PV)
    em = em.squeeze()
    J_L = q * torch.sum(em * nb_B_e * IQE) * (lambda_i[1] - lambda_i[0])
    J_0 = q * torch.sum(nb_B_PV*IQE) * (lambda_i[1] - lambda_i[0])

    V_oc = (k_B*T_PV/q)*torch.log(J_L/J_0+1)
    t = torch.linspace(0, 1, 100)
    V = t * V_oc

    J = J_L-J_0*(torch.exp(q*V/(k_B*T_PV))-1)
    P = V*J

    return torch.max(P)


def power_ratio(lambda_i, emissivity_dataset, T_emitter, E_g_PV):
    emissivity = emissivity_dataset.squeeze()
    P_emit = torch.sum(emissivity*Blackbody(lambda_i, T_emitter)
                       ) * (lambda_i[1] - lambda_i[0])
    IQE_PV = IQE(lambda_i, E_g_PV)
    JV_PV = JV(emissivity, IQE_PV, lambda_i)

    FOM = JV_PV / P_emit
    return FOM

def get_unique_log_dir(base_dir, config):
    """
    Generates a unique directory name based on the config.
    If a directory with the same config exists, appends a suffix.
    """
    # Serialize the config to a JSON string
    config_str = json.dumps(config, sort_keys=True)
    # Create a hash of the config for unique identification
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    base_name = f"def-{config_hash}"
    log_dir = os.path.join(base_dir, base_name)

    # If directory exists, append a suffix
    suffix = 1
    while os.path.exists(log_dir):
        log_dir = os.path.join(base_dir, f"{base_name}_{suffix}")
        suffix += 1

    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def fourier_series_analysis(
        values: np.ndarray,
        K: int = 10,
        period: float | int = 1.0,
        plot: bool = True
) -> list[tuple[int, complex]]:
    """
    Compute Fourier-series coefficients cₖ and (optionally) plot a truncated
    reconstruction of a piece-wise-constant function whose fundamental period
    is *period* (not necessarily 1).

    Parameters
    ----------
    values : 1-D array_like
        Constant values on the sub-intervals that partition one period.
        The j-th entry is the value on
        [j·Δx , (j+1)·Δx),  where Δx = period / N  and N = len(values).
    K : int, optional
        Maximum |k| harmonic to keep (-K … K).  Default 10.
    period : float | int, optional
        Fundamental period L (>0).  Default 1.0.
    plot : bool, optional
        Whether to plot the real & imaginary parts of the original function
        and its |k|≤K approximation on [0, period).

    Returns
    -------
    fourier_amplitudes : list[(int, complex)]
        [(k, cₖ) for k = -K … K]
    """
    # ------------------------------------------------------------
    #  Pre-compute common scalars
    # ------------------------------------------------------------
    values = np.asarray(values, dtype=complex)
    N       = len(values)
    L       = float(period)                 # make sure it's a float
    Δx      = L / N                         # interval width

    # ------------------------------------------------------------
    #  Fourier coefficients
    # ------------------------------------------------------------
    c = np.empty(2 * K + 1, dtype=complex)

    for k in range(-K, K + 1):
        idx = k + K                         # 0-based index into c

        # k = 0 → average value over one period
        if k == 0:
            c[idx] = np.mean(values)
            continue

        # k ≠ 0 → exact integral over each sub-interval
        sum_val = 0.0 + 0.0j
        for j, a_j in enumerate(values):
            x_start = j * Δx
            x_end   = (j + 1) * Δx
            term = (
                np.exp(-1j * 2 * np.pi * k * x_start / L)
                - np.exp(-1j * 2 * np.pi * k * x_end   / L)
            )
            sum_val += a_j * term

        c[idx] = sum_val / (1j * 2 * np.pi * k)        # 1/(i2πk) Σ …
    # ------------------------------------------------------------
    #  Helper to evaluate truncated series
    # ------------------------------------------------------------
    def series_approx(x: np.ndarray) -> np.ndarray:
        f = np.zeros_like(x, dtype=complex)
        for k in range(-K, K + 1):
            f += c[k + K] * np.exp(1j * 2 * np.pi * k * x / L)
        return f

    # ------------------------------------------------------------
    #  Helper for the original piece-wise function
    # ------------------------------------------------------------
    def piecewise(x: np.ndarray) -> np.ndarray:
        x_mod = np.mod(x, L)                # fold into one period
        j     = np.floor(x_mod / Δx).astype(int)
        return values[j]

    # ------------------------------------------------------------
    #  Optional plotting
    # ------------------------------------------------------------
    if plot:
        x_dense  = np.linspace(0, L, 1000, endpoint=False)
        y_orig   = piecewise(x_dense)
        y_series = series_approx(x_dense)

        plt.figure(figsize=(10, 10))

        # Real part
        plt.subplot(2, 1, 1)
        plt.plot(x_dense, np.real(y_orig),   label='Original (Real)',
                 drawstyle='steps-post', linewidth=2)
        plt.plot(x_dense, np.real(y_series), label=f'Fourier |k|≤{K} (Real)',
                 linestyle='--', linewidth=2)
        plt.title(f'Fourier Series Approximation – Real part (period = {L})')
        plt.ylabel('Re f(x)')
        plt.legend(); plt.grid(True)

        # Imaginary part
        plt.subplot(2, 1, 2)
        plt.plot(x_dense, np.imag(y_orig),   label='Original (Imag)',
                 drawstyle='steps-post', linewidth=2)
        plt.plot(x_dense, np.imag(y_series), label=f'Fourier |k|≤{K} (Imag)',
                 linestyle='--', linewidth=2)
        plt.title(f'Fourier Series Approximation – Imag part (period = {L})')
        plt.xlabel('x'); plt.ylabel('Im f(x)')
        plt.legend(); plt.grid(True)

        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------
    #  Package results
    # ------------------------------------------------------------
    fourier_amplitudes = [(k, c[k + K]) for k in range(-K, K + 1)]
    return fourier_amplitudes


def _fourier_series_analysis(values, K=10, plot=True):
    """
    Computes the complex Fourier series coefficients for a piecewise constant function,
    produces an approximation using harmonics k = -K, ..., K, and optionally plots both 
    the original function and its Fourier series approximation (separately for the real 
    and imaginary parts).

    Parameters:
      values : array_like
          A 1D array of complex (or real) numbers defining the function on intervals [j, j+1)
          for j = 0,1,...,N-1. The function is assumed to be periodic with period L = N.
      K : int, optional
          The maximum positive harmonic to include. The function computes coefficients for
          k = -K, -K+1, ..., 0, ..., K. Default is 10.
      plot : bool, optional
          If True, generates two subplots comparing the real and imaginary parts of the original
          function and its Fourier series approximation.

    Returns:
      fourier_amplitudes : list of tuples
          A list of (k, c_k) pairs where c_k is the complex Fourier coefficient for harmonic k.
    """
    # Determine period
    N = len(values)
    L = N  # period

    # Helper function to compute complex Fourier coefficients c_k for k=-K,...,K.
    def compute_fourier_coefficients_complex(values, K):
        # index: 0 => k=-K, index K => k=0, index 2K => k=K
        c = np.zeros(2*K+1, dtype=complex)
        for k in range(-K, K+1):
            idx = k + K  # shift index: k=-K -> 0, ..., k=0 -> K, k=K -> 2K
            if k == 0:
                c[idx] = (1.0 / L) * np.sum(values)
            else:
                sum_val = 0.0
                for j in range(N):
                    # Integral over [j, j+1] of exp(-i*2π*k*x/L) dx:
                    term = np.exp(-1j * 2 * np.pi * k * j / L) - \
                        np.exp(-1j * 2 * np.pi * k * (j+1) / L)
                    sum_val += values[j] * term
                c[idx] = (1.0 / (1j * 2 * np.pi * k)) * sum_val
        return c

    # Helper function to evaluate the Fourier series approximation at points x.
    def fourier_series_complex_approx(x, c, K, period):
        f_approx = np.zeros_like(x, dtype=complex)
        for k in range(-K, K+1):
            idx = k + K
            f_approx += c[idx] * np.exp(1j * 2 * np.pi * k * x / period)
        return f_approx

    # Helper function to evaluate the original piecewise function at x.
    def piecewise_function(x, values):
        x_mod = np.mod(x, N)
        y = np.zeros_like(x, dtype=complex)
        for j in range(N):
            y[(x_mod >= j) & (x_mod < j+1)] = values[j]
        return y

    # Compute Fourier coefficients
    c = compute_fourier_coefficients_complex(values, K)

    # Create a dense x-axis over one period for evaluation and plotting.
    x = np.linspace(0, L, 1000)
    y_orig = piecewise_function(x, values)
    y_fourier = fourier_series_complex_approx(x, c, K, L)

    # Optionally produce plots for the real and imaginary parts.
    if plot:
        plt.figure(figsize=(10, 10))
        # Plot real parts
        plt.subplot(2, 1, 1)
        plt.plot(x, np.real(y_orig), label="Original (Real)",
                 drawstyle='steps-post', linewidth=2)
        plt.plot(x, np.real(
            y_fourier), label=f"Fourier Approx (Real) (|k| ≤ {K})", linestyle='--', linewidth=2)
        plt.title("Fourier Series Approximation (Real Part)")
        plt.xlabel("x")
        plt.ylabel("Real f(x)")
        plt.legend()
        plt.grid(True)
        # Plot imaginary parts
        plt.subplot(2, 1, 2)
        plt.plot(x, np.imag(y_orig), label="Original (Imaginary)",
                 drawstyle='steps-post', linewidth=2)
        plt.plot(x, np.imag(
            y_fourier), label=f"Fourier Approx (Imag) (|k| ≤ {K})", linestyle='--', linewidth=2)
        plt.title("Fourier Series Approximation (Imaginary Part)")
        plt.xlabel("x")
        plt.ylabel("Imaginary f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Prepare list of Fourier amplitudes as (harmonic, amplitude) pairs.
    fourier_amplitudes = [(k, c[k + K]) for k in range(-K, K+1)]
    return fourier_amplitudes


def get_harmonic_index(basis_set, harmonic):
    """
    Finds the 1-indexed position of the tuple (harmonic, 0) in the basis set.

    Parameters:
        basis_set: List of tuples from GetBasisSet()
        harmonic: The harmonic index (0-indexed) to search for

    Returns:
        The 1-indexed position of (harmonic, 0) in the basis set,
        or None if not found
    """
    for idx, (i, j) in enumerate(basis_set):
        if i == harmonic and j == 0:
            return idx + 1  # Convert to 1-indexed

    return None  # Return None if the harmonic is not found

def create_step_excitation_fft(basis, step_values, num_harmonics=25, x_shift=0, initial_phase=0, amplitude=1.0, plot_fourier=True):
    """
    Creates excitations that replicate a step function pattern in the x-direction using NumPy's FFT.
    
    Parameters:
        basis: The basis set from Ssample.GetBasisSet()
        step_values: Array of values defining the step function (e.g., [1, -1])
        num_harmonics: Maximum harmonic to include in the Fourier series
        x_shift: Shifts the pattern in the x-direction (0-1 range)
        initial_phase: Initial phase angle in radians for the entire pattern
        amplitude: Overall amplitude scaling factor
        plot_fourier: Whether to plot the Fourier series approximation
        
    Returns:
        List of excitation tuples ready for SetExcitationExterior
    """
    # Convert to numpy array if not already
    step_values = np.array(step_values, dtype=complex)
    N = len(step_values)
    
    # Compute FFT
    fft_result = np.fft.fft(step_values) / N  # Normalize by N
    
    # Create initial phase factor
    phase_factor = np.exp(1j * initial_phase)
    
    # Prepare excitations list
    excitations = []
    
    # Plot Fourier series approximation if requested
    if plot_fourier:
        # Create a dense x-axis over one period for plotting
        x = np.linspace(0, 1, 1000)
        y_orig = np.zeros_like(x, dtype=complex)
        for j in range(N):
            mask = (x >= j/N) & (x < (j+1)/N)
            y_orig[mask] = step_values[j]
        
        # Reconstruct signal from selected harmonics
        y_fourier = np.zeros_like(x, dtype=complex)
        for k in range(-num_harmonics, num_harmonics+1):
            k_idx = k % N  # Handle negative indices correctly
            if k_idx < 0:
                k_idx += N
            
            # Skip harmonics that exceed our limit
            if abs(k) > num_harmonics:
                continue
                
            # Apply the proper coefficient
            coef = fft_result[k_idx]
            y_fourier += coef * np.exp(1j * 2 * np.pi * k * x)
        
        # Plot real and imaginary parts
        plt.figure(figsize=(10, 10))
        
        # Real part
        plt.subplot(2, 1, 1)
        plt.plot(x, np.real(y_orig), label="Original (Real)", 
                 drawstyle='steps-post', linewidth=2)
        plt.plot(x, np.real(y_fourier), label=f"Fourier Approx (Real) (|k| ≤ {num_harmonics})", 
                 linestyle='--', linewidth=2)
        plt.title("Fourier Series Approximation (Real Part)")
        plt.xlabel("x")
        plt.ylabel("Real f(x)")
        plt.legend()
        plt.grid(True)
        
        # Imaginary part
        plt.subplot(2, 1, 2)
        plt.plot(x, np.imag(y_orig), label="Original (Imaginary)", 
                 drawstyle='steps-post', linewidth=2)
        plt.plot(x, np.imag(y_fourier), label=f"Fourier Approx (Imag) (|k| ≤ {num_harmonics})", 
                 linestyle='--', linewidth=2)
        plt.title("Fourier Series Approximation (Imaginary Part)")
        plt.xlabel("x")
        plt.ylabel("Imaginary f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Process Fourier coefficients and create excitations
    for k in range(-num_harmonics, num_harmonics+1):
        # Find correct index in FFT result array
        k_idx = k % N
        if k_idx < 0:
            k_idx += N
            
        # Get coefficient
        c_k = fft_result[k_idx]
        
        # Find the corresponding index in the S4 basis
        idx = get_harmonic_index(basis, k)
        
        if idx is not None:
            # Apply x-shift if needed
            if x_shift != 0:
                # For a shift in real space, multiply by exp(-i*k*x_shift)
                shift_phase = np.exp(-1j * 2 * np.pi * k * x_shift)
                c_k *= shift_phase
            
            # Apply global phase and amplitude scaling
            final_amplitude = amplitude * phase_factor * c_k
            
            # Add to excitations
            excitations.append((idx, b'x', final_amplitude))
    
    return excitations


def create_step_excitation(basis, step_values, num_harmonics=25, x_shift=0, initial_phase=0, amplitude=1.0, period = 1.0, plot_fourier=True):
    """
    Creates excitations that replicate a step function pattern in the x-direction.

    Parameters:
        basis: The basis set from Ssample.GetBasisSet()
        step_values: Array of values defining the step function (e.g., [1, -1])
        num_harmonics: Maximum harmonic to include in the Fourier series
        x_shift: Shifts the pattern in the x-direction (0-1 range)
        initial_phase: Initial phase angle in radians for the entire pattern
        amplitude: Overall amplitude scaling factor
        plot_fourier: Whether to plot the Fourier series approximation

    Returns:
        List of excitation tuples ready for SetExcitationExterior
    """
    # Get Fourier coefficients for the step function
    fourier_coeffs = fourier_series_analysis(
        step_values, num_harmonics, period = period, plot=plot_fourier)

    # Create initial phase factor
    phase_factor = np.exp(1j * initial_phase)

    # Prepare excitations list
    excitations = []

    # Process each Fourier coefficient
    for k, c_k in fourier_coeffs:
        # Find the corresponding index in the S4 basis
        idx = get_harmonic_index(basis, k)

        if idx is not None:
            # Apply x-shift if needed
            if x_shift != 0:
                # For a shift in real space, multiply by exp(-i*k*x_shift)
                # Assuming period = 1 for normalized coordinates
                shift_phase = np.exp(-1j * 2 * np.pi * k * x_shift / period)
                c_k *= shift_phase

            # Apply global phase and amplitude scaling
            final_amplitude = amplitude * phase_factor * c_k

            # Add to excitations
            excitations.append((idx, b'y', final_amplitude))

    return excitations


def home_directory():
    return os.path.expanduser("~/tpv-shaping")


h = 6.626070e-34  # Js Planck's constant
c = 2.997925e8  # m/s speed of light
k_B = 1.380649e-23  # J/K Boltzmann constant
q = 1.602176e-19  # C elementary charge
e_0 = 8.8541878128e-12
T_PV = 300  # K PV temperature
T_e = 2073.15  # K emitter temperature

 
def Blackbody(lambda_i, T):
    return (2*h*c**2) / ((np.exp((h*c)/(k_B*T*lambda_i*1e-6))-1)*lambda_i**5)*1e14


def nb_B(lambda_i, T):
    return (2*c) / ((np.exp((h*c)/(k_B*T*lambda_i*1e-6))-1)*lambda_i**4)*1e8


def write_to_temp_file(line, max_lines=1000):
    temp_file = Path(home_directory()) / 'transmitted_power_log.txt'

    # Read existing lines if file exists
    lines = []
    if temp_file.exists():
        with open(temp_file, 'r') as f:
            lines = f.readlines()

    # Add new line and keep only last max_lines
    lines.append(line + '\n')
    lines = lines[-max_lines:]

    # Write back to file
    with open(temp_file, 'w') as f:
        f.writelines(lines)


def load_config_in_dir(dir_path):
    """Try to load config.json from dir_path. Return dict if successful, else None."""
    config_path = os.path.join(dir_path, 'config.json')
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def read_live_gradient_scale(default: float):
    try:
        with open(f'{home_directory()}/.LIVE_GRADIENT_SCALE.txt', 'r') as f:
            return float(f.read().strip())
    except FileNotFoundError:
        return default  # Use default if file doesn't exist


def replace_nan_with_neighbors(arr):
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            left = arr[i-1] if i > 0 else arr[i+1]
            right = arr[i+1] if i < len(arr)-1 else arr[i-1]
            arr[i] = (left + right) / 2
    return arr


def quar(mat: npt.ArrayLike) -> np.ndarray:
    assert mat.shape[0] == mat.shape[1]
    assert mat.shape[0] % 2 == 0
    mid = int(mat.shape[0] / 2)
    return np.array([mat[:mid, :mid], mat[:mid, mid:], mat[mid:, :mid], mat[mid:, mid:]])

# TODO vectorize for GPU use and give errors


def manual_matmul(A, B, threshold=None, dtype=None):
    if dtype is None:
        dtype = np.cdouble
    else:
        dtype = np.dtype(dtype)
    """
    Manually performs matrix multiplication between two NumPy arrays A and B.

    Parameters:
    - A: NumPy array of shape (m, n)
    - B: NumPy array of shape (n, p)

    Returns:
    - result: NumPy array of shape (m, p) resulting from A x B
    """

    # Get the dimensions of the input matrices
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape

    # Check if the matrices can be multiplied
    if a_cols != b_rows:
        raise ValueError("Incompatible dimensions for matrix multiplication.")

    # Initialize the result matrix with zeros
    # This class has worked so far to elim half-zero errors
    result = np.zeros((a_rows, b_cols), dtype=dtype)

    # Perform the matrix multiplication manually
    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(a_cols):  # or range(b_rows)
                result[i, j] += A[i, k] * B[k, j]
    if threshold is not None:
        result[np.abs(result) < threshold] *= 0
    return result


def read_boolean_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        _, boolean_values = line.split(",", 1)
        boolean_list = ast.literal_eval(boolean_values.strip())
        int_list = [int(value) for value in boolean_list]
        data.append(int_list)
    data_array = np.array(data, dtype=int)
    return data_array


def parse_data(lines):
    data_arrays = []
    for line in lines:
        if line.strip():  # This checks if the line is not empty
            array_str = line.split(",", 1)[1].strip().strip("[]")
            array = np.array(list(map(float, array_str.split(","))))
            data_arrays.append(array)
    return np.array(data_arrays)


def read_data(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            value = line.strip().split(", ")[1]
            data.append(float(value))
    return np.array(data, dtype=object)


def plotLens(array):
    plt.figure(figsize=(10, 5))
    plt.step(range(len(array)), array, where="post")
    plt.show()
