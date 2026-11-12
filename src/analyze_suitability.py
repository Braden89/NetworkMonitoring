"""Read all source CSVs in bounded batches; save observations without training."""
import json
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import quote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

from src.data_loader import CACHE, ROOT, inspect_dataset
from src.quality import identify_label


def analyze():
    metadata = inspect_dataset()
    files = [f for f in metadata['files'] if f['path'].endswith('.csv')]
    print(f"Inspecting {len(files)} CSVs, {sum(f['bytes'] for f in files):,} source bytes", flush=True)
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'metadata': metadata,
              'scope': 'Full scan of every source CSV, in batches of 25000; no preprocessing or training.',
              'files': [], 'classes': Counter(), 'missing': Counter(), 'infinite': Counter(),
              'negative': Counter(), 'invalid_numeric': Counter()}
    hashes = []
    minima, maxima = {}, {}
    first_columns = None
    repeated_header_equal = True
    for file in files:
        url = f"https://huggingface.co/datasets/{metadata['dataset']}/resolve/{metadata['revision']}/{quote(file['path'])}"
        stream = load_dataset('csv', data_files={'train': url}, split='train',
                              streaming=True, cache_dir=str(CACHE))
        entry = {'file': file['path'], 'records': 0, 'classes': Counter(), 'rows_with_nonfinite': 0}
        for batch in stream.iter(batch_size=25000):
            frame = pd.DataFrame(batch)
            label = identify_label(frame)
            if first_columns is None:
                first_columns = list(frame.columns)
                result['columns'] = first_columns
                result['first_batch_types'] = {c: str(t) for c, t in frame.dtypes.items()}
            if list(frame.columns) != first_columns:
                raise ValueError(f"Schema differs in {file['path']}")
            counts = frame[label].fillna('<MISSING>').value_counts().to_dict()
            entry['records'] += len(frame)
            entry['classes'].update(counts)
            result['classes'].update(counts)
            result['missing'].update({c: int(v) for c, v in frame.isna().sum().items() if v})
            numeric = frame.drop(columns=label).apply(pd.to_numeric, errors='coerce')
            invalid = numeric.isna() & frame.drop(columns=label).notna()
            for key, values in [('invalid_numeric', invalid.sum()),
                                ('infinite', np.isinf(numeric).sum()),
                                ('negative', (numeric < 0).sum())]:
                result[key].update({c: int(v) for c, v in values.items() if v})
            entry['rows_with_nonfinite'] += int((~np.isfinite(numeric)).any(axis=1).sum())
            for c in numeric:
                lo, hi = numeric[c].min(), numeric[c].max()
                if pd.notna(lo): minima[c] = min(minima.get(c, lo), lo)
                if pd.notna(hi): maxima[c] = max(maxima.get(c, hi), hi)
            names = {c.strip(): c for c in frame.columns}
            a, b = names.get('Fwd Header Length'), names.get('Fwd Header Length.1')
            repeated_header_equal &= bool(a and b and frame[a].equals(frame[b]))
            # Stable string representation makes fingerprints consistent across batch dtype inference.
            hashes.append(pd.util.hash_pandas_object(frame.astype(str), index=False).to_numpy())
        result['files'].append(entry)
        print(f"{entry['file']}: {entry['records']:,} rows; {dict(entry['classes'])}", flush=True)
    fingerprints = np.concatenate(hashes)
    result['records'] = sum(f['records'] for f in result['files'])
    result['features'] = len(first_columns) - 1
    result['duplicate_fingerprints_beyond_first'] = int(len(fingerprints) - len(np.unique(fingerprints)))
    result['duplicate_method'] = '64-bit row fingerprints of string values, including label; collision risk and dtype formatting mean this is an estimate, not exact row equality.'
    result['constant_numeric_features'] = {c: float(minima[c]) for c in minima if minima[c] == maxima[c] and not result['missing'][c]}
    result['repeated_header_columns_equal'] = repeated_header_equal
    result['rows_with_nonfinite'] = sum(f['rows_with_nonfinite'] for f in result['files'])
    folder = ROOT / 'reports'
    (folder / 'dataset_suitability.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    counts = pd.Series(result['classes']).sort_values()
    fig, ax = plt.subplots(figsize=(11, 7))
    counts.plot.barh(ax=ax, logx=True, color='#2878a0')
    ax.set(title='CIC-IDS2017: all source records', xlabel='Flow records (log scale)', ylabel='Source label')
    fig.tight_layout()
    fig.savefig(folder / 'attack_distribution.png', dpi=160)
    plt.close(fig)
    write_assessment(result)
    print(f"Saved full-scan observations: {result['records']:,} rows", flush=True)


def write_assessment(result):
    """Pair measured counts with a clearly identified project recommendation."""
    total = result['records']
    benign = result['classes'].get('BENIGN', 0)
    lines = ["# CIC-IDS2017 project suitability", "",
             "## Recommendation", "",
             "Use this dataset for the graduate project's offline binary baseline and selected attack classification. Treat live detection as a separate validation milestone: CSV performance alone cannot establish real-network effectiveness.",
             "", "## Actual inspection", "",
             f"Revision: `{result['metadata']['revision']}`. Generated UTC: {result['generated_utc']}.",
             result['scope'],
             f"Examined {total:,} records across {len(result['files'])} files, with {result['features']} features and one label column.",
             f"BENIGN: {benign:,} ({benign / total:.2%}); attack-labeled: {total-benign:,} ({(total-benign)/total:.2%}). An always-BENIGN classifier would score {benign/total:.2%} accuracy.",
             "", "| Observed source label | Records | Share |", "|---|---:|---:|"]
    for label, n in sorted(result['classes'].items(), key=lambda item: -item[1]):
        lines.append(f"| {label} | {n:,} | {n/total:.4%} |")
    lines += ["", "![Full dataset class distribution](attack_distribution.png)", "",
              "## Coverage by file", "", "| File | Records | Labels |", "|---|---:|---|"]
    for f in result['files']:
        lines.append(f"| {f['file']} | {f['records']:,} | {', '.join(f['classes'])} |")
    lines += ["", "## Quality findings", "",
              f"- Missing cells: {sum(result['missing'].values()):,}; infinite numeric cells: {sum(result['infinite'].values()):,}.",
              f"- Rows with any nonfinite or unparseable feature: {result['rows_with_nonfinite']:,}.",
              f"- Numeric parse failures among nonmissing features: {sum(result['invalid_numeric'].values()):,}.",
              f"- Duplicate row fingerprint estimate beyond first occurrence: {result['duplicate_fingerprints_beyond_first']:,}. {result['duplicate_method']}",
              f"- Constant numeric features: {result['constant_numeric_features']}.",
              f"- Fwd Header Length and Fwd Header Length.1 identical in every batch: {result['repeated_header_columns_equal']}.",
              "- Negative values need feature-specific interpretation; for example, window sizes may use sentinels. Do not discard all negative values blindly.",
              "- Original headers and labels are retained. Review header whitespace and corrupted characters in web-attack labels before encoding.",
              "", "Detailed per-column counts and source metadata are in `dataset_suitability.json`.",
              "", "## Implications for this project", "",
              "1. The observed DDoS, DoS, PortScan, FTP-Patator, and SSH-Patator labels cover the requested attack families. Start with benign versus attack; assess individual classes only where enough examples exist. Heartbleed, Infiltration, and SQL Injection are too scarce for strong standalone claims.",
              "2. Report per-class precision/recall, macro F1, confusion matrices, and benign false-positive rate. Accuracy is insufficient because of imbalance. Sample across files and labels; the initial 10,000-row sample missed all attacks.",
              "3. Resolve duplicate leakage before future splits. Preserve source filename as provenance, not as a model input. This mirror lacks timestamps and source/destination IPs, so exact temporal or host-group splitting is unavailable. File/day holdouts can omit entire attack classes; inspect coverage before choosing an evaluation design.",
              "4. These are flow-level features, not packet-level examples. Scapy capture alone cannot reproduce flow duration, directional statistics, interarrival times, active/idle windows, and TCP features. Define a compatible extractor, direction rules, units, and flow timeout; validate with PCAP replay before live use. Partial flows also differ from completed training flows.",
              "5. This is a controlled 2017 benchmark. Neither high offline accuracy nor these labels establishes performance on present-day traffic or unseen attacks. Published research identifies extraction and labeling problems; assess those limitations in the final report.",
              "", "## Source context (separate from measured statistics)", "",
              "- [Original UNB dataset description](https://www.unb.ca/cic/datasets/ids-2017.html): five-day collection, attack schedule, and CICFlowMeter-derived flow features.",
              "- [Engelen et al., troubleshooting study and documentation](https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html): flow construction, labeling, flag counting, and redundant-feature problems. This analysis does not establish that the Hugging Face mirror fixes those problems.",
              "", "No preprocessing, splitting, model fitting, or live capture was implemented.", ""]
    (ROOT / 'reports' / 'dataset_suitability.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    analyze()
