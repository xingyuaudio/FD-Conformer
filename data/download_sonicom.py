"""Download SONICOM P0001-P0200 FreeFieldCompMinPhase 48 kHz SOFA files."""
import argparse
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent / "raw"
URL_TEMPLATE = (
    "https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/"
    "{pid}/HRTF/HRTF/48kHz/{pid}_FreeFieldCompMinPhase_48kHz.sofa"
)


def download_sonicom(output_dir: Path = RAW_DIR) -> list[str]:
    """Download files explicitly; return failed participant IDs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i in range(1, 201):
        pid = f"P{i:04d}"
        save_path = output_dir / f"{pid}.sofa"
        partial_path = output_dir / f"{pid}.sofa.part"
        if save_path.exists() and save_path.stat().st_size > 0:
            print(f"Already exists: {save_path.name}")
            continue

        try:
            with requests.get(URL_TEMPLATE.format(pid=pid), stream=True, timeout=30) as response:
                response.raise_for_status()
                size = 0
                with partial_path.open("wb") as handle:
                    for chunk in tqdm(
                        response.iter_content(chunk_size=8192), desc=pid, unit="chunk"
                    ):
                        if chunk:
                            handle.write(chunk)
                            size += len(chunk)
                if size == 0:
                    raise ValueError("Empty download.")
                expected = response.headers.get("Content-Length")
                if expected and not response.headers.get("Content-Encoding"):
                    if size != int(expected):
                        raise ValueError("Downloaded size differs from Content-Length.")
                partial_path.replace(save_path)
        except (requests.RequestException, OSError, ValueError) as error:
            partial_path.unlink(missing_ok=True)
            failed.append(pid)
            print(f"Download failed for {pid}: {error}")
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)
    failed = download_sonicom(args.output_dir)
    if failed:
        print(f"Failed subjects ({len(failed)}): {', '.join(failed)}")
        return 1
    print(f"SOFA files are available in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
