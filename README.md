# Hardware RBF Pen Digits System

For installation, operation, troubleshooting, data contracts, and agent handoff,
use the primary Chinese guide: [README_cn.md](README_cn.md). It is stored as
UTF-8 with BOM for reliable display in Windows PowerShell, WPS, and common editors.

The project has exactly four Python `main()` entry points:

- `train_hardware_model.py` trains the 16×16 hardware Gaussian model.
- `ui.py` opens the unified scientific GUI for separate sample-saving and
  manually triggered hardware-inference experiments.
- `collect_pen_digits.py` runs collection only and appends 16-value samples.
- `run_hardware_inference.py` continuously processes new rows and predicts digits.

The inference process is a restart-safe, append-only pipeline:

```text
16 raw values (0..100)
-> normalize and quantize to the 256 differential levels
-> append a 16-value differential row
-> choose one measured physical cycle from the 400-cycle response bank
-> look up all 16 Group responses for each quantized dimension
-> append a dimension-major 256-value hardware row
-> atomically overwrite the fixed sorted 17-column hardware experiment table
-> classify with checkpoints/weights.pt
-> append the prediction and detailed report
```

Operational events and shared-file synchronization are appended to `SURF/app.log`. Agent implementation progress is recorded separately in `SURF/agent.log`.

Runtime CSV files are stored outside this repository in `SURF/runtime/`. Every
digit processed by Experiment 2 atomically overwrites
`SURF/runtime/hardware_experiments/pen_digits_hardware_experiment.csv`.
Its first column, `differential_level_index`, is the integer zero-based position
in `differential_levels.csv` (`0` through `255`); its next 16 columns are the
corresponding hardware block. Equal levels are merged only among the current
digit's 16 dimensions, and rows are sorted ascending. Data from different digits
is never accumulated: only the latest processed digit remains in the fixed file.
This keeps generated samples and reports out of the `rbf-hardware` Git history.

## Run

From the `SURF` directory on Windows, launch the unified GUI:

```powershell
.\venv\Scripts\Activate.ps1
python .\rbf-hardware\ui.py
```

The GUI owns collection and inference in one process. Do not run the headless
continuous inference entry against the same runtime CSV files at the same time.
The interface starts at a size appropriate for the current screen. Wide windows
use a two-column research console; portrait or narrow windows automatically stack
collection above inference and enable vertical page scrolling. The drawing pad
scales with either layout while preserving its logical coordinates, so resizing
or rotating the window does not change the generated 16-value sample.
`Experiment 1 · Save` appends only the drawn sample. `Experiment 2 · Infer`
processes pending rows only when clicked (or when `F5` is pressed); saving and
external-row detection never start inference automatically. Experiment 2 remains
disabled until at least one pipeline stage has pending work.
If the window is closed during Experiment 2, it stays open until the inference
worker finishes all staged CSV writes, then exits safely.
All visible GUI text is English. The Hardware Output panel restores and displays
the latest 256-value response in scientific notation with horizontal scrolling;
the fixed experiment CSV is overwritten while Experiment 2 processes each sample.

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

Training displays `tqdm` progress for every completed cross-validation
parameter tuple and for the four final model-building stages.

Use `--sampling-mode mean` for deterministic verification. The default `empirical` mode draws one of 400 measured physical cycles per saved digit and uses that same cycle for all 16 dimensions. After exact level-and-Group lookup, independent multiplicative jitter is applied: up to ±5% for small responses, decreasing linearly to ±1% at each Group's measured 95th-percentile magnitude. This avoids copying source values while preserving signs, tails, channel correlation, and cycle drift without a Gaussian assumption.

The checkpoint consumes the generated 256-value hardware row, never the original 16-value input. The runtime response bank contains 256 levels × 400 cycles × 16 Groups from the supplied physical-device workbook. The former Gaussian parameter CSV is retained only as a historical artifact and is no longer loaded by inference.

Do not keep runtime CSV files open in WPS or Excel during inference because those
applications may lock the files on Windows. Continuous inference logs the lock and
retries automatically; processing resumes from the incomplete stage after the file
is closed.

## Package layout

`src/rbf_hardware/` is organized by responsibility: `configuration`, `data`, `inference`, `infrastructure`, `modeling`, `reporting`, `training`, `ui`, and `utilities`. Library modules contain no command-line entry points.

See [README_cn.md](README_cn.md) for the complete data contract, runtime files, commands, and verification results.
