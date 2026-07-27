# Hardware RBF Pen Digits System

The project has exactly three Python `main()` entry points:

- `train_hardware_model.py` trains the 16×16 hardware Gaussian model.
- `collect_pen_digits.py` appends 16-value mouse-drawn samples.
- `run_hardware_inference.py` continuously processes new rows and predicts digits.

The inference process is a restart-safe, append-only pipeline:

```text
16 raw values (0..100)
-> normalize and quantize to the 256 differential levels
-> append a 16-value differential row
-> choose one measured physical cycle from the 400-cycle response bank
-> look up all 16 Group responses for each quantized dimension
-> append a dimension-major 256-value hardware row
-> create this digit's sorted 17-column differential/hardware aggregate
-> classify with checkpoints/weights.pt
-> append the prediction and detailed report
```

Operational events and shared-file synchronization are appended to `SURF/app.log`. Agent implementation progress is recorded separately in `SURF/agent.log`.

Runtime CSV files are stored outside this repository in `SURF/runtime/`. Every saved digit creates a uniquely named CSV under `SURF/runtime/hardware_experiments/`. Its first column, `differential_level_index`, is the integer zero-based position in `differential_levels.csv` (`0` through `255`); its next 16 columns are the corresponding hardware block. Equal levels are merged only among that digit's 16 dimensions, and rows are sorted ascending. Data from different digits is never accumulated into one experiment table. This keeps generated samples and reports out of the `rbf-hardware` Git history.

## Run

From the `SURF` directory on Windows:

```powershell
.\venv\Scripts\Activate.ps1
python .\rbf-hardware\collect_pen_digits.py
python .\rbf-hardware\run_hardware_inference.py
python .\rbf-hardware\train_hardware_model.py
```

Process currently available rows once:

```powershell
python .\rbf-hardware\run_hardware_inference.py --once
```

Use `--sampling-mode mean` for deterministic verification. The default `empirical` mode draws one of 400 measured physical cycles per saved digit and uses that same cycle for all 16 dimensions. After exact level-and-Group lookup, independent multiplicative jitter is applied: up to ±5% for small responses, decreasing linearly to ±1% at each Group's measured 95th-percentile magnitude. This avoids copying source values while preserving signs, tails, channel correlation, and cycle drift without a Gaussian assumption.

The checkpoint consumes the generated 256-value hardware row, never the original 16-value input. The runtime response bank contains 256 levels × 400 cycles × 16 Groups from the supplied physical-device workbook. The former Gaussian parameter CSV is retained only as a historical artifact and is no longer loaded by inference.

Do not keep runtime CSV files open in WPS or Excel during inference because those
applications may lock the files on Windows. Continuous inference logs the lock and
retries automatically; processing resumes from the incomplete stage after the file
is closed.

## Package layout

`src/rbf_hardware/` is organized by responsibility: `configuration`, `data`, `inference`, `infrastructure`, `modeling`, `reporting`, `training`, `ui`, and `utilities`. Library modules contain no command-line entry points.

See [README_cn.md](README_cn.md) for the complete data contract, runtime files, commands, and verification results.
