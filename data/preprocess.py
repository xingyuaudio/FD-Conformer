"""Convert SONICOM P0001-P0200 SOFA HRIRs to H_log (200, 793, 2, 106)."""
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.fft import fft
import sofar
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
OUTPUT_PATH = DATA_DIR / "sonicom.npz"


def linear2db(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, eps))


def preprocess(
    raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH
) -> np.ndarray:
    """Validate the paper's grid/FFT assumptions and save float32 dB magnitudes."""
    raw_dir, output_path = Path(raw_dir), Path(output_path)
    sofa_paths = sorted(
        path for path in raw_dir.glob("P0*.sofa")
        if path.stem[1:5].isdigit() and 1 <= int(path.stem[1:5]) <= 200
    )
    if len(sofa_paths) != 200:
        raise ValueError("Expected 200 subjects (P0001-P0200).")
    values = []
    for index, path in enumerate(tqdm(sofa_paths)):
        if int(path.stem[1:5]) != index + 1:
            raise ValueError(f"Subject order mismatch: {path}")
        with contextlib.redirect_stdout(io.StringIO()):
            sofa = sofar.read_sofa(str(path))
        hrir = np.asarray(sofa.Data_IR)
        if hrir.shape != (793, 2, 256):
            raise ValueError(f"{path.name}: expected HRIR shape (793, 2, 256), got {hrir.shape}.")
        sample_rate = np.asarray(sofa.Data_SamplingRate)
        if sample_rate.size == 0 or not np.allclose(sample_rate, 48000, rtol=0, atol=1e-6):
            raise ValueError(f"{path.name}: expected sampling rate 48000 Hz.")
        if not np.isfinite(hrir).all():
            raise ValueError(f"{path.name}: HRIR contains non-finite values.")

        # Original computation: remove DC, retain FFT bins 1 through 106.
        spectrum = np.abs(fft(hrir, axis=-1))
        selected = spectrum[..., 1:107]
        values.append(linear2db(selected).astype(np.float32))

    H_log = np.stack(values, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, H_log=H_log)
    print("Saved:", output_path)
    print("H_log shape:", H_log.shape)
    return H_log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    try:
        preprocess(args.raw_dir, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
