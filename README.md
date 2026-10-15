# Exploring Frequency-Domain Feature Modeling for HRTF Magnitude Upsampling

Official PyTorch implementation of the paper:

> **Exploring Frequency-Domain Feature Modeling for HRTF Magnitude Upsampling**

Accepted for publication in The Journal of the Acoustical Society of America (JASA).
The final version is being prepared. Preprint: [arXiv:2602.11670](https://arxiv.org/abs/2602.11670).

HRTF upsampling is inherently a spatial reconstruction task: recovering HRTFs
at unmeasured directions from sparse spatial measurements. This work complements
spatial reconstruction by organizing HRTF features along the frequency axis,
enabling explicit modeling of cross-frequency feature correlations. We
systematically evaluate different frequency-domain modeling strategies
(MLP, Conv1D, Dilated Conv, Conformer, etc.) for HRTF magnitude upsampling
on the SONICOM dataset.

---

# 1. Dataset Preparation

Run the commands below from the repository root. Training/evaluation paths in
YAML are relative to the current working directory. The data preparation scripts
default to this repository's `data/raw/` and `data/sonicom.npz` paths.

We use the **SONICOM HRTF dataset** (200 subjects, 793 directions).

---

## Option A — Download raw SOFA files and preprocess

### Step 1: Download SONICOM dataset

You can batch-download `.sofa` files using:

```bash
python data/download_sonicom.py
```

This will download files:

```
data/raw/P0001.sofa
...
data/raw/P0200.sofa
```

---

### Step 2: Preprocess to generate `data/sonicom.npz`

```bash
python data/preprocess.py
```

This will generate:

```
data/sonicom.npz
```

The saved file contains:

```
H_log: (200, 793, 2, 106)
```

- 200 subjects  
- 793 directions  
- 2 ears  
- 106 frequency bins  
- Log-magnitude in dB  

---

## Option B — Use a preprocessed `.npz`

```
data/sonicom.npz
```

If you already have this file, you can skip raw download and preprocessing.
Its `H_log` array must have shape `(200, 793, 2, 106)` in float32 dB magnitudes.

### Data and pretrained checkpoint downloads

The processed data and pretrained weights are not bundled in this repository.
Their download links will be added after the complete artifact release is
published; the table below describes the intended local placement.

| Artifact | Download | Local path |
|---|---|---|
| SONICOM processed data | Pending publication | `data/sonicom.npz` |
| 3-point Conformer | Pending publication | `checkpoints/m3/best_model.pt` |
| 5-point Conformer | Pending publication | `checkpoints/m5/best_model.pt` |
| 19-point Conformer | Pending publication | `checkpoints/m19/best_model.pt` |
| 100-point Conformer | Pending publication | `checkpoints/m100/best_model.pt` |

The existing [Zenodo v1.0 archive](https://zenodo.org/records/21134348) contains
the source snapshot only. [The all-versions DOI](https://doi.org/10.5281/zenodo.21134347)
will resolve to the latest published version. Neither link currently provides
the processed data or pretrained checkpoints.

Future downloads will include matching inference configurations and checksums.
Set `data.n_measurements` and the model settings to match the selected weights.
Sparse indices in `src/datasets/sparse_sets.py` are zero-based and ordered; do
not reorder them. Retain the original SONICOM attribution and license when
sharing derived data.

---

# 2. Training

Training is controlled by a YAML configuration file.

Edit:

```
configs/default.yaml
```

Set the number of sparse measurements:

```yaml
data:
  n_measurements: 3   # choose from {3, 5, 19, 100}
```

---

## Run training

```bash
python src/train.py --config configs/default.yaml
```

Training will:

- Load SONICOM dataset  
- Split subjects into train/test  
- Train model  
- Save best model to:

```
runs/<exp_name>_<timestamp>/best_model.pt
```

---

# 3. Evaluation

The checkpoint selected during training minimizes the sample-weighted training
shape loss. Test metrics are monitoring results and do not select the checkpoint.

After training, evaluate using:

```bash
python src/eval.py \
    --config configs/default.yaml \
    --ckpt runs/<run_name>/best_model.pt
```

Evaluation reports:

- LSD (Log Spectral Distortion) on unseen directions  
- Broadband ILD MAE  

Illustrative output (not reference benchmark scores):

```
Evaluation Results
==============================
Sparse measurements : 3
Train LSD (unseen)  : 4.12 dB
Test  LSD (unseen)  : 4.85 dB
Train ILD MAE       : 1.24 dB
Test  ILD MAE       : 1.37 dB
==============================
```

---

# 4. Model Variants

Set model variant in YAML:

```yaml
model:
  variant: conformer   # options:
                       # mlp
                       # conv
                       # dilated_b2
                       # conformer
                       # conformer_wo_conv
```

All variants share the same architecture except for the frequency modeling block.

---

# 5. Environment

Tested with:

```
Python 3.10
PyTorch 2.6.0
```

Install the pinned PyTorch runtime first. For CUDA 12.4 and a compatible driver:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

For CPU-only use, replace the first command with:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

See the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/).



---

# 6. Citation

If you use this code, please cite:

```
@article{chen2026exploring,
  title={Exploring Frequency-Domain Feature Modeling for HRTF Magnitude Upsampling},
  author={Chen, Xingyu and Bi, Hanwen and Ma, Fei and Zhao, Sipei and Cheng, Eva and Burnett, Ian S},
  journal={arXiv preprint arXiv:2602.11670},
  year={2026}
}
```




## Small regression checks

With the listed dependencies installed, run:

```bash
python -m unittest discover -s tests -v
```

These checks cover uneven batches, safe tensor loading, sparse-index copies,
shared float32 dataset storage, and data-script import/validation behavior.
