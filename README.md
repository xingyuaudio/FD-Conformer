# Exploring Frequency-Domain Feature Modeling for HRTF Magnitude Upsampling

Official PyTorch implementation of the paper:

> **Exploring Frequency-Domain Feature Modeling for HRTF Magnitude Upsampling**

This repository investigates different frequency-domain modeling strategies
(MLP, Conv1D, Dilated Conv, Conformer, etc.) for sparse-to-dense HRTF
magnitude upsampling on the SONICOM dataset.

---

# 1. Dataset Preparation

We use the **SONICOM HRTF dataset** (200 subjects, 793 directions).

---

## Option A — Download raw SOFA files and preprocess

### Step 1: Download SONICOM dataset

You can batch-download `.sofa` files using:

```bash
python tools/download_sonicom.py
```

This will download files:

```
raw/P0001.sofa
...
raw/P0200.sofa
```

---

### Step 2: Preprocess to generate `sonicom.npz`

```bash
python tools/preprocess_sonicom.py
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

## Option B — Directly use provided `.npz`

```
data/sonicom.npz
```

You can skip the raw download and preprocessing steps.

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
runs/sonicom_<variant>_<date>/best_model.pt
```

---

# 3. Evaluation

After training, evaluate using:

```bash
python src/eval.py \
    --config configs/default.yaml \
    --ckpt runs/<run_name>/best_model.pt
```

Evaluation reports:

- LSD (Log Spectral Distortion) on unseen directions  
- Broadband ILD MAE  

Example output:

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

Install dependencies:

```bash
pip install -r requirements.txt
```
### Install PyTorch (recommended)

For CUDA 12.4:

```bash
conda install pytorch=2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```



---

# 6. Citation

If you use this code, please cite:

```
https://arxiv.org/pdf/2602.11670
```



