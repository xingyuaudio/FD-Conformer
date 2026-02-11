"""
Preprocess SONICOM SOFA files (P0001–P0200)

Input:
    raw/P0001_FreeFieldCompMinPhase_48kHz.sofa
    ...
    raw/P0200_FreeFieldCompMinPhase_48kHz.sofa

Output:
    sonicom.npz

Saved variable:
    H_log  (200, 793, 2, 106)
"""

from pathlib import Path
import numpy as np
from scipy.fft import fft
from tqdm import tqdm
import sofar
import contextlib
import io


# ===== paths =====
RAW_DIR = Path("raw/")
OUTPUT_PATH = Path("sonicom.npz")


def linear2db(x, eps=1e-8):
    return 20.0 * np.log10(np.maximum(x, eps))


# ===== collect sofa files =====
sofa_paths = sorted(RAW_DIR.glob("P0*.sofa"))

# Keep P0001–P0200 only
sofa_paths = [
    p for p in sofa_paths
    if 1 <= int(p.stem[1:5]) <= 200
]

assert len(sofa_paths) == 200, "Expected 200 subjects (P0001–P0200)"

H_log_list = []

for idx, path in enumerate(tqdm(sofa_paths)):
    pid = int(path.stem[1:5])
    assert pid - 1 == idx, f"Subject order mismatch: {path}"

    # Suppress sofar print messages
    with contextlib.redirect_stdout(io.StringIO()):
        sofa = sofar.read_sofa(str(path))

    # HRIR shape: (793, 2, T)
    hrir = sofa.Data_IR

    # FFT along time axis
    spec = np.abs(fft(hrir, axis=-1))

    # Keep positive frequencies
    spec = spec[..., : hrir.shape[-1] // 2 + 1]

    # Remove DC bin and keep 106 bins
    spec_lin = spec[:, :, 1:107]  # (793, 2, 106)

    H_log = linear2db(spec_lin)

    H_log_list.append(H_log.astype(np.float32))


H_log_array = np.stack(H_log_list, axis=0)  # (200, 793, 2, 106)

np.savez(
    OUTPUT_PATH,
    H_log=H_log_array,
)

print("Saved:", OUTPUT_PATH)
print("H_log shape:", H_log_array.shape)
