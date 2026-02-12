import os
import requests
from tqdm import tqdm

"""
download_sonicom.py
────────────────────
Batch download script for the SONICOM HRTF dataset (.sofa files).

This script downloads FreeFieldCompMinPhase 48kHz HRTFs
for subjects P0001 to P0325 and saves them into the local
'data/raw' directory.

Note:
- Make sure you have access permission to the SONICOM server.
- The script skips files that already exist.
"""

# Create local dataset directory if it does not exist
os.makedirs("raw/", exist_ok=True)

# URL template for each participant ID
url_template = (
    "https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/"
    "{pid}/HRTF/HRTF/48kHz/{pid}_FreeFieldCompMinPhase_48kHz.sofa"
)

# Download .sofa files for P0001 to P0201
for i in range(1, 201):
    pid = f"P{i:04d}"
    url = url_template.format(pid=pid)
    filename = f"{pid}.sofa"
    save_path = os.path.join("raw/", filename)

    # Skip download if file already exists
    if os.path.exists(save_path):
        print(f"Already exists: {filename}")
        continue

    try:
        print(f"Downloading: {filename}")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(chunk_size=8192),
                    desc=filename,
                    unit="KB",
                ):
                    if chunk:
                        f.write(chunk)

    except Exception as e:
        print(f"Download failed for {filename}: {e}")
