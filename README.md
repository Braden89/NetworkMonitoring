# Real-Time Network Intrusion Detection Using Machine Learning

Graduate ML project: explore labeled network flows, then develop intrusion classification and eventually a real-time monitoring application over five weeks. **Only Phase 1 is implemented.** No preprocessing, training, packet capture, backend, or dashboard is included.

Dataset: [muzom/CIC-IDS2017](https://huggingface.co/datasets/muzom/CIC-IDS2017). The source repository contains eight CSV files. The loader inspects repository sizes, revision, configuration, splits, and feature metadata before streaming records through Hugging Face `datasets`. Dataset usage is subject to the source license; review it before redistribution.

## WSL Ubuntu setup

Run from this repository in a WSL Ubuntu terminal (Python 3.12):

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
cd /mnt/c/Users/brade/Desktop/Classwork/MachineLearning/NetworkMonitoring
python3.12 -m venv ~/.venvs/networkmonitoring
source ~/.venvs/networkmonitoring/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name networkmonitoring --display-name "NetworkMonitoring (Python 3.12)"
python -m src.data_loader --inspect-only
python -m src.data_loader --records 10000 --seed 42 --buffer-size 10000
jupyter lab notebooks/01_data_exploration.ipynb
```

In VS Code, use **WSL: Reopen Folder in WSL** from the Command Palette (install Microsoft's WSL extension if that command is unavailable). Confirm the lower-left corner shows WSL Ubuntu. Open `notebooks/01_data_exploration.ipynb`, click **Select Kernel**, and choose **NetworkMonitoring (Python 3.12)**. If needed, choose **Python Environments** and enter `/home/<your-wsl-username>/.venvs/networkmonitoring/bin/python`. Install Microsoft's Python and Jupyter extensions in WSL if prompted. Run the cells in order or select **Run All**.

If virtual environment creation fails, install `python3.12-venv` using the command above. Network errors require working HTTPS access to Hugging Face and its dataset storage endpoints; retry `python -m src.data_loader --inspect-only` after restoring access. Dependency installation failures: retry `python -m pip install -r requirements.txt` and inspect the reported package/network error.

The environment is kept on WSL's Linux filesystem because installation under `/mnt/c` can be very slow. In each new WSL terminal, activate it before running project commands:

```bash
source ~/.venvs/networkmonitoring/bin/activate
python -c "import sys; print(sys.executable)"
```

The persistent environment was installed and validated on November 11, 2026. Both focused quality tests (`python -m unittest discover -s tests -v`), dependency consistency (`python -m pip check`), and complete notebook execution using the registered `networkmonitoring` kernel passed. Python and Jupyter extensions are installed in WSL. The old project-local `.venv` is incomplete and should not be selected. The generated report describes 10,000 observed records; all sampled labels were BENIGN, illustrating the sampling limitation rather than the absence of attacks in the dataset.

## Exploration and reports

The notebook inspects metadata first and defaults to 10,000 streamed records. Adjust `RECORDS`, `SEED`, and `BUFFER_SIZE` to expand exploration. The CLI and notebook both generate `reports/data_quality_summary.md` from actual observations. The report includes class distribution, missing and infinite values, duplicate rows, numeric parsing issues, constant columns, and negative values requiring domain review. The report separates sample statistics from repository metadata. Original column names and labels are retained.

The buffered shuffle is reproducible for a fixed dataset revision, seed, buffer size, and package environment. It is **not a uniform or stratified sample of all files** and may miss rare attacks or show strong ordering bias. Increasing the record count and buffer improves coverage but does not guarantee all categories. Streaming limits rows held in memory but can prefetch extra bytes; it is not a strict download-byte cap. `--inspect-only` fetches metadata without preparing the full CSV dataset. If source metadata omits counts or types, those remain unknown until records are inspected. No full-dataset quality claims are inferred from this sample.

Cache files live in ignored `data/raw/hf_cache/`; large data, virtual environments, and model artifacts stay out of Git. Dependency ranges are compatibility bounds, not an exact environment lock. The generated report records the dataset commit so subsequent inspection can pin it with `inspect_dataset(revision=...)`.

## Layout and status

- `src/data_loader.py`: repository inspection, bounded streaming, CLI.
- `src/quality.py`: reusable observation and Markdown report functions.
- `notebooks/01_data_exploration.ipynb`: interactive EDA and plots.
- `data/raw/`, `data/processed/`, `models/`: reserved artifact directories.
- `reports/data_quality_summary.md`: observed data quality and provenance.

Future phases require approval: Phase 2 handles missing/infinite values, label encoding, leakage-aware splits, and preprocessing pipelines; Phase 3 compares Logistic Regression and Random Forest with binary/multiclass metrics; Phase 4 develops compatible flow extraction and live/replay classification; Phase 5 adds FastAPI, an optional dashboard, and a final evaluation/report. Future module files are deferred until their phase begins.
