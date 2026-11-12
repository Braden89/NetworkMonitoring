# Real-Time Network Intrusion Detection Using Machine Learning

Graduate ML project: explore labeled network flows, then develop intrusion classification and eventually a real-time monitoring application over five weeks. **Phases 1 and 2 plus the Phase 3 binary baseline are implemented.** Dataset exploration, reusable preprocessing, and binary classification are available. Multiclass training, packet capture, backend, and dashboard remain future work.

Dataset: [muzom/CIC-IDS2017](https://huggingface.co/datasets/muzom/CIC-IDS2017). The source repository contains eight CSV files. The loader inspects repository sizes, revision, configuration, splits, and feature metadata before streaming records through Hugging Face `datasets`. Dataset usage is subject to the source license; review it before redistribution.

## WSL Ubuntu setup

Run from this repository in a WSL Ubuntu terminal (Python 3.12):

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
cd /path/to/NetworkMonitoring
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

Validate your active environment with `python -m pip check` and `python -m unittest discover -s tests -v`. The generated report describes actual sampled records, not full-dataset statistics.

## Exploration and reports

The notebook inspects metadata first and defaults to 10,000 streamed records. Adjust `RECORDS`, `SEED`, and `BUFFER_SIZE` to expand exploration. The CLI and notebook both generate `reports/data_quality_summary.md` from actual observations. The report includes class distribution, missing and infinite values, duplicate rows, numeric parsing issues, constant columns, and negative values requiring domain review. The report separates sample statistics from repository metadata. Original column names and labels are retained.

The buffered shuffle is reproducible for a fixed dataset revision, seed, buffer size, and package environment. It is **not a uniform or stratified sample of all files** and may miss rare attacks or show strong ordering bias. Increasing the record count and buffer improves coverage but does not guarantee all categories. Streaming limits rows held in memory but can prefetch extra bytes; it is not a strict download-byte cap. `--inspect-only` fetches metadata without preparing the full CSV dataset. If source metadata omits counts or types, those remain unknown until records are inspected. No full-dataset quality claims are inferred from this sample.

Cache files live in ignored `data/raw/hf_cache/`; large data, virtual environments, and model artifacts stay out of Git. Dependency ranges are compatibility bounds, not an exact environment lock. The generated report records the dataset commit so subsequent inspection can pin it with `python -m src.data_loader --revision <commit> --records 10000` or `inspect_dataset(revision=...)`.

## Layout and status

- `src/data_loader.py`: repository inspection, bounded streaming, CLI.
- `src/quality.py`: reusable observation and Markdown report functions.
- `notebooks/01_data_exploration.ipynb`: interactive EDA and plots.
- `data/raw/`, `data/processed/`, `models/`: reserved artifact directories.
- `reports/data_quality_summary.md`: observed data quality and provenance.

Phase 2 now handles missing/infinite values, label encoding, duplicate-aware splits, and reusable preprocessing. Future phases require approval: Phase 3 compares Logistic Regression and Random Forest with binary/multiclass metrics; Phase 4 develops compatible flow extraction and live/replay classification; Phase 5 adds FastAPI, an optional dashboard, and a final evaluation/report. Future module files are deferred until their phase begins.

## Dataset suitability analysis

To inspect every source CSV in batches of 25,000 records, run:

```bash
python -m src.analyze_suitability
```

This reads roughly 885 MB of source CSV data over the network, but does not hold the full dataset in memory. It retains compact row fingerprints for a duplicate estimate. Allow several minutes depending on network and CPU speed. The analysis preserves source labels and does not preprocess data or train a model.

Outputs: `reports/dataset_suitability.md` (assessment), `reports/dataset_suitability.json` (measured counts and provenance), and `reports/attack_distribution.png` (full-scan class distribution). The original `data_quality_summary.md` remains a separate sample-level report.

## Phase 2: preprocessing and splits

Use the **NetworkMonitoring (Python 3.12)** notebook kernel. Install all dependencies together in its environment:

```bash
source ~/.venvs/networkmonitoring/bin/activate
python -m pip install -r requirements.txt
python -m src.prepare_data --per-class 5000 --seed 42 --revision b33f3052398ffd63463ce3120fec2acbb68e2ff6
jupyter lab notebooks/02_preprocessing.ipynb
```

The first preparation pass streams all eight source CSVs (about 885 MB), retaining up to 5,000 uniformly sampled rows per raw label using a seeded random-priority reservoir. It takes several minutes. Later runs reuse `data/raw/phase2_sample.parquet` and its provenance JSON when parameters match. `--refresh-sample` forces a new scan; `--revision` pins the source commit. Omitting revision on a cached run keeps the recorded cached revision. Reproducibility also requires the same package versions and batch size, recorded in the output. Larger `--per-class` values increase sample memory use and runtime. Sampling happens before deduplication, so some classes retain fewer than the requested cap.

This class-capped sample is a **development benchmark**, not an estimate of natural traffic prevalence. The Phase 1 10,000-row sample is not used for preprocessing.

The pipeline:

1. Retains source filename and row number as provenance, excluded from model features.
2. Trims headers/labels and normalizes observed web-attack separators. Drops the verified redundant `Fwd Header Length.1` feature.
3. Converts invalid numeric strings and infinities to missing values. Converts `-1` only in the two initial TCP-window columns to missing; other negatives remain pending domain review.
4. Removes conflicting-label feature groups and exact repeated cleaned feature rows before splitting.
5. Creates seeded approximately 70/15/15 splits within each class. Classes with fewer than three unique records are training-only and flagged. These are random record splits, not independent host/session/time holdouts.
6. Fits median imputation, constant-feature removal, and standard scaling on training only. Training-all-missing features become zero and are removed as constants. Binary targets use BENIGN=0 and attack=1; multiclass encoding is fitted on training labels.
7. Removes transformed duplicates from validation/test, including overlaps introduced by cleaning, imputation, or constant removal. Reports any resulting loss of class coverage.

The [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) explains why fitting transforms only on training matters. Future cross-validation must fit a fresh pipeline **inside each fold** using the saved raw training data.

Artifacts (large/generated data and pipeline objects are ignored by Git):

- `data/processed/phase2/{train,validation,test}.npz`: `X`, `y_binary`, and `y_multiclass`, with matching row order.
- `data/processed/phase2/{train,validation,test}_raw.parquet`: original feature values, normalized labels/headers, and provenance for each retained split. Use these for future fold-local fitting.
- `data/processed/phase2/summary.json`: source revision, sample parameters, versions, label mapping, feature names, and split counts.
- `models/phase2_preprocessor.joblib`: fitted preprocessing pipeline, label encoder, and expected raw feature list. Load only trusted local joblib artifacts.
- `reports/phase2_preprocessing_summary.md`: readable preparation results and limitations.
- `notebooks/02_preprocessing.ipynb`: preparation walkthrough, class coverage, saved pipeline, and validation.

Example loading for a later phase:

```python
import numpy as np
with np.load("data/processed/phase2/train.npz") as data:
    X_train = data["X"]
    y_train = data["y_binary"]
```

Validation:

```bash
python -m unittest discover -s tests -v
python -m pip check
```

No classifier is trained in Phase 2. Review rare-class support and remaining session/time leakage risks before Phase 3.

## Phase 3: binary baseline

```bash
source ~/.venvs/networkmonitoring/bin/activate
python -m src.train
jupyter lab notebooks/03_binary_baseline.ipynb
```

The training command fits Logistic Regression and Random Forest on the fixed Phase 2 training split, compares validation macro F1 (lower benign false-positive rate breaks ties), and scores only the selected model on test. Both use balanced class weights and a fixed default decision threshold of 0.5. Nonconverged candidates cannot win. A majority-class dummy baseline provides context. No hyperparameter or threshold search is performed, and validation data is not used to refit the winner.

Subsequent runs return existing results after checking the seed and Phase 2 artifact hashes; they do not retrain or rescore test. Changed Phase 2 artifacts require a deliberately designed new experiment, rather than silently reusing stale results. The notebook displays saved results and demonstrates inference using validation examples. If results are missing, run the CLI first.

- `reports/phase3_binary_baseline.md`: comparison, selected-model test metrics, and limitations.
- `reports/binary_baseline_metrics.json`: metrics, parameters, warnings, timings, version, and input hashes.
- `reports/binary_test_confusion_matrix.png`: selected model's test confusion matrix.
- `models/binary_detector.joblib`: preprocessing plus classifier, schema, and binary label mapping (ignored by Git).
- `src/predict.py`: offline predictions from compatible raw CSV/Parquet flow features.

```bash
python -m src.predict data/processed/phase2/validation_raw.parquet --output data/processed/phase3/validation_predictions.csv
```

Prediction output is BENIGN or ATTACK plus an uncalibrated `attack_score`; it does not identify individual attacks. Only load trusted local joblib artifacts. The input must contain compatible flow measurements, not raw packets. These results describe an attack-heavy, class-capped development benchmark, not production precision or live-network performance. Do not tune against the recorded test score.

Model references: [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
