"""Report observed quality issues without cleaning or transforming the dataset."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from src.data_loader import ROOT


def identify_label(frame):
    candidates = [c for c in frame.columns if str(c).strip().casefold() in {"label", "class", "attack", "attack_type"}]
    if len(candidates) != 1:
        raise ValueError(f"Label column is ambiguous; inspect columns: {list(frame.columns)}")
    return candidates[0]


def quality_summary(frame):
    label = identify_label(frame)
    numeric = frame.select_dtypes(include="number")
    infinite = pd.Series(np.isinf(numeric.to_numpy()).sum(axis=0), index=numeric.columns)
    invalid = {}
    for col in frame.columns.drop(label):
        if not pd.api.types.is_numeric_dtype(frame[col]):
            invalid[col] = int((frame[col].notna() & pd.to_numeric(frame[col], errors="coerce").isna()).sum())
    return {"label": label, "records": len(frame), "features": len(frame.columns) - 1,
            "classes": frame[label].fillna("<MISSING>").value_counts().to_dict(),
            "missing": frame.isna().sum().to_dict(),
            "duplicates": int(frame.duplicated().sum()), "infinite": infinite.to_dict(),
            "nonnumeric_feature_values": invalid,
            "negative_numeric_values": (numeric < 0).sum().to_dict(),
            "constant_columns": frame.columns[frame.nunique(dropna=False) <= 1].tolist(),
            "whitespace_column_names": [c for c in frame.columns if str(c) != str(c).strip()]}


def write_report(frame, metadata, seed=42, buffer_size=10000, split="train", path=None):
    summary = quality_summary(frame)
    path = ROOT / "reports" / "data_quality_summary.md" if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data quality summary", "", f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
             "", "## Scope", "", f"Source: https://huggingface.co/datasets/{metadata['dataset']}",
             f"Revision: `{metadata['revision']}`; split: `{split}`; seed: {seed}; shuffle buffer: {buffer_size}.",
             "All quality statistics below describe only the examined sample. Buffered streaming is not a uniform global sample; file/time ordering can omit attack categories. No full-dataset quality claims are made.",
             "", f"Records examined: {summary['records']}; feature columns (excluding label): {summary['features']}.",
             f"Label column: `{summary['label']}`; duplicate rows beyond first occurrence: {summary['duplicates']}.",
             "", "## Observed attack categories and class distribution", "", "| Category | Count | Sample % |", "|---|---:|---:|"]
    for key, count in summary['classes'].items():
        lines.append(f"| {key} | {count} | {100 * count / len(frame):.3f} |")
    for title, key in [("Missing values", "missing"), ("Infinite numerical values", "infinite"),
                       ("Nonnumeric values in non-label text columns", "nonnumeric_feature_values"),
                       ("Negative numerical values (review domain meaning)", "negative_numeric_values")]:
        values = {k: int(v) for k, v in summary[key].items() if v}
        lines += ["", f"## {title}", "", f"Total: {sum(values.values())}", "", "```json", json.dumps(values, indent=2), "```"]
    lines += ["", "## Potential preprocessing problems", "",
              f"Constant columns in sample: {summary['constant_columns']}",
              f"Column names with outer whitespace: {summary['whitespace_column_names']}",
              "Review missing/infinite values, numeric parsing, negative sentinel values, class imbalance, duplicated flows, and redundant features before training. Constant columns may vary outside this sample. Duplicate counting uses all columns including the label; conflicting labels require separate investigation. Preserve temporal/file provenance when designing future splits to prevent leakage. No rows or columns were cleaned here.",
              "", "## Observed sample columns and types", "", "```json",
              json.dumps({c: str(t) for c, t in frame.dtypes.items()}, indent=2), "```",
              "", "## Repository metadata (not computed sample statistics)", "",
              "Null counts and empty feature metadata mean the source builder did not supply them; sample types are listed above.",
              "", "```json", json.dumps(metadata, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
