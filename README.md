# Hardware RBF Post-Layer Classifier

This project trains the fully connected classifier after the 256-dimensional RBF features produced by the hardware device. It mirrors the `pen_digits` classifier head in `rbf-network-benchmark`, while replacing the PC-computed Gaussian RBF features with measured hardware outputs.

`train.py` is the only training entry point. Implementation details are separated into small modules under `src/rbf_hardware/`.

## Network and training method

The benchmark path is:

```text
16 quantized pen coordinates
→ 256 Gaussian RBF features
→ append one intercept feature
→ 257 × 10 multi-output ridge-regression weights
→ argmax over digit classes 0–9
```

The classifier uses the same closed-form ridge objective as the benchmark:

```text
W = (Phi.T @ Phi + alpha * I)^(-1) @ Phi.T @ one_hot(y)
```

The intercept is not regularized. This is a closed-form ridge solution; there is no optimizer, learning rate, batch size, or epoch count in the benchmark.

The PC benchmark value `ridge_alpha: 0.001` remains the baseline candidate. Hardware features are much more correlated than the PC Gaussian centers, so the default run selects the effective alpha by deterministic 5-fold stratified cross-validation on `train_out` only. The current data selects `alpha: 0.3`. No test labels participate in selection.

The device currents are approximately `-5e-9` to `4.36e-6`, while ideal Gaussian activations are on an order-one non-negative scale. The default configuration therefore:

1. clamps tiny negative device noise to zero;
2. fits one global maximum on `train_out.csv` only;
3. applies that training scale to both train and test features;
4. saves the fitted scale and policy inside `weights.pt`.

This train-only fit avoids test leakage. A second train-only, out-of-fold calibration selects a small score bias for priority class 8. It changes only the existing fully connected bias vector; it does not add a layer. The current data selects `class_8_bias: 0.07` while satisfying the configured training-fold recall floor.

## Why the original result was low

The original fixed-alpha run produced 100% training accuracy but only 75.60% test accuracy. The normalized design matrix has 257 columns, only 439 training rows, a condition number near 4,596, and strongly correlated hardware responses. With `alpha: 0.001`, the maximum absolute combined weight was approximately 9.24, indicating an unstable, overfitted solution.

Class 8 also has the greatest within-class spread. Its nearest class centroid is class 0, and the 8-to-0 centroid distance is only about 46% of class 8's within-class RMS spread. The classifier consequently under-predicted class 8 and frequently mapped it to 0.

The train-only alpha search reduces the maximum absolute weight to about 0.665. Out-of-fold class-bias calibration then corrects class 8 under-prediction. Current measured results are:

| Run | Test accuracy | Macro-F1 | Class 8 recall |
|---|---:|---:|---:|
| Fixed benchmark alpha `0.001` | 75.60% | 74.99% | 54.48% |
| CV alpha `0.3` | 84.73% | 84.30% | 68.82% |
| CV alpha + train-only class 8 bias (previous 439/3,000 split) | **84.90%** | **84.55%** | **75.27%** |
| Current swapped 3,000/439 split | **84.28%** | **84.13%** | **73.81%** |

The last two rows use different test samples and are not a direct like-for-like comparison. The current swapped run selects `alpha: 0.3` and `class_8_bias: 0.1`; its train-only five-fold accuracy is 86.86%.

## Dataset contract

All CSV files have no header; the final column is the integer label.

| File | Shape | Meaning | Use |
|---|---:|---|---|
| `dataset/csv/train_in.csv` | 10,553 × 17 | 16 pre-RBF inputs + label | Audit the first 3,000 training labels |
| `dataset/csv/train_out.csv` | 3,000 × 257 | 256 hardware RBF outputs + label | Classifier training |
| `dataset/csv/test_in.csv` | 439 × 17 | 16 pre-RBF inputs + label | Test-label audit |
| `dataset/csv/test_out.csv` | 439 × 257 | 256 hardware RBF outputs + label | Classifier evaluation |

The post-RBF classifier must use `train_out.csv` and `test_out.csv` as features. A `*_in.csv` file has only 16 features and cannot be passed directly to a 256-input fully connected layer. The loader cross-checks hardware labels against the first 3,000 rows of `train_in.csv` and all 439 rows of `test_in.csv`. The remaining 7,553 training-reference inputs do not yet have hardware outputs.

The loader also rejects exact feature rows shared by train and test. The current isolation audit finds zero shared feature rows and zero shared labeled rows. File SHA-256 fingerprints and the audit result are stored in `metrics.json`.

The supplied `test_out.csv` has been inspected during iterative development. Its 84.90% result is exactly reproducible for these 3,000 rows, but it is not a pristine, never-observed blind-test estimate. For a publication-grade final claim, freeze the current checkpoint and collect hardware outputs for new samples (or for the remaining 7,553 `test_in` rows) before evaluating once.

## Project structure

```text
rbf-hardware/
├── config.yaml                  # every runtime/training/output option
├── train.py                     # only training entry point with main()
├── requirements.txt
├── README.md
├── README_cn.md
├── dataset/
│   ├── csv/
│   │   ├── train_in.csv
│   │   ├── train_out.csv
│   │   ├── test_in.csv
│   │   └── test_out.csv
│   └── origin_xslx/
└── src/rbf_hardware/
    ├── config.py                # configuration validation and path resolution
    ├── data.py                  # CSV/schema/label checks and train-fit scaling
    ├── model.py                 # benchmark-compatible ridge classifier
    ├── metrics.py               # metrics, reports, CSV and SVG confusion matrix
    ├── logging_utils.py         # shared terminal + app.log logging
    └── training.py              # training and artifact orchestration
```

## Run in the current workspace

From the `SURF` parent directory, use the existing virtual environment:

```powershell
.\venv\Scripts\python.exe -u .\rbf-hardware\train.py --config .\rbf-hardware\config.yaml
```

Or from `rbf-hardware`:

```powershell
..\venv\Scripts\python.exe -u train.py --config config.yaml
```

The `-u` option makes terminal logging unbuffered.

To watch the same shared log in a second VS Code PowerShell terminal:

```powershell
Get-Content ..\app.log -Wait -Encoding UTF8
```

The terminal and `SURF/app.log` receive the same timestamped messages. `[LOOK]` and `[CHANGE]` markers expose the data inspection, preprocessing/training change, and post-change evaluation loop.

## Deploy on another PC

Copy the complete `rbf-hardware` directory, including `dataset/csv`, into a writable parent directory:

```text
<workspace>/
└── rbf-hardware/
```

Do not copy an existing virtual environment between PCs. Create a new one on the target machine.

### Windows PowerShell

Python 3.10–3.12, 64-bit, is recommended.

```powershell
cd <workspace>\rbf-hardware
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -u train.py --config config.yaml
```

These commands invoke the virtual-environment interpreter directly, so PowerShell script-execution policy does not need to be changed.

In a second VS Code terminal:

```powershell
cd <workspace>\rbf-hardware
Get-Content ..\app.log -Wait -Encoding UTF8
```

### Linux or macOS

```bash
cd <workspace>/rbf-hardware
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -u train.py --config config.yaml
```

In a second terminal:

```bash
cd <workspace>/rbf-hardware
tail -F ../app.log
```

The default requirements install a portable PyTorch build selected by `pip`. For a specific CUDA build, install the PyTorch wheel recommended for the target CUDA driver first, then install the remaining requirements. Set `runtime.device` to `auto`, `cpu`, or `cuda` in `config.yaml`.

## Outputs

All run artifacts are written outside the project, under the parent workspace:

```text
<workspace>/output/rbf_hardware/pen_digits_hardware_fc_<timestamp>/
├── weights.pt
├── metrics.json
├── predictions.csv
├── confusion_matrix.csv
├── confusion_matrix.txt
├── confusion_matrix.svg
├── classification_report.txt
├── alpha_search.csv
├── class_bias_search.csv
└── config.yaml
```

The shared live log is `<workspace>/app.log`.

`weights.pt` contains CPU tensors so it can be loaded on a machine without CUDA. It includes:

- `state_dict.weight`: shape `[10, 256]`;
- `state_dict.bias`: shape `[10]`;
- `combined_weights`: benchmark-compatible shape `[257, 10]`;
- class labels, baseline and selected ridge settings, selected class bias, fitted hardware scale, metrics, source paths, and a config snapshot.

`metrics.json` records dataset/runtime metadata, preprocessing, split-isolation hashes/audits, evaluation-protocol status, the complete train-only alpha and class-bias searches, numerical diagnostics, per-class recall, and final train/test metrics. `alpha_search.csv` and `class_bias_search.csv` are compact versions of the selection tables.

Example checkpoint inspection:

```python
import torch

checkpoint = torch.load("weights.pt", map_location="cpu", weights_only=False)
weight = checkpoint["state_dict"]["weight"]
bias = checkpoint["state_dict"]["bias"]
scale = checkpoint["preprocessing"]["scale"]
```

Apply the saved `negative_policy` and `scale` before using `weight` and `bias` for inference.

## Configuration

All adjustable settings are in `config.yaml`:

- `paths`: portable workspace-relative data, output, and log paths;
- `data`: delimiter, encoding, feature counts, labels, and reference checks;
- `preprocessing`: negative-noise policy and train-fit scaling;
- `classifier`: ridge structure, benchmark baseline alpha, train-only alpha selection, and class-score calibration;
- `runtime`: device, dtype, random seed, deterministic execution;
- `output`: run naming and artifact filenames;
- `logging`: level and file mode;
- `diagnostics`: generalization-gap and per-class recall warning thresholds.

Relative paths are resolved from the project/workspace layout, not from the shell's current directory.

## Troubleshooting

- **“17 columns … cannot be sent to the 256-input classifier”**: `test_in.csv` or `train_in.csv` was incorrectly configured as a post-RBF feature file. Use `test_out.csv` or `train_out.csv`.
- **Reference row difference**: only the first 3,000 of 10,553 `train_in` rows currently have corresponding hardware outputs. This is expected for the supplied files.
- **CUDA requested but unavailable**: set `runtime.device: cpu`, or install a CUDA-compatible PyTorch build and driver.
- **Permission denied under the parent workspace**: the parent directory must be writable because outputs and `app.log` are intentionally outside `rbf-hardware`.
- **CSV parse or column error**: keep the supplied files headerless and retain the final label column.
