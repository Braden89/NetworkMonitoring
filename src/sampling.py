"""Bounded, seeded per-label sampling over every pinned source CSV."""
from collections import Counter
from urllib.parse import quote

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.data_loader import CACHE, inspect_dataset
from src.quality import identify_label


def retain_by_priority(previous, batch, per_class):
    """Keep lowest random priorities: a uniform reservoir within each raw label."""
    merged = pd.concat([previous, batch], ignore_index=True) if previous is not None else batch
    return merged.nsmallest(per_class, '_priority').reset_index(drop=True)


def sample_all_files(per_class=5000, seed=42, revision=None, batch_size=25000):
    if per_class < 1 or batch_size < 1:
        raise ValueError('per_class and batch_size must be positive')
    metadata = inspect_dataset(revision)
    files = sorted(f['path'] for f in metadata['files'] if f['path'].endswith('.csv'))
    if not files:
        raise ValueError('No source CSV files found in inspected repository')
    rng = np.random.default_rng(seed)
    reservoirs, counts, file_counts = {}, Counter(), {}
    columns = None
    for filename in files:
        url = f"https://huggingface.co/datasets/{metadata['dataset']}/resolve/{metadata['revision']}/{quote(filename)}"
        stream = load_dataset('csv', data_files={'train': url}, split='train',
                              streaming=True, cache_dir=str(CACHE))
        offset = 0
        for batch in stream.iter(batch_size=batch_size):
            frame = pd.DataFrame(batch)
            if columns is None:
                columns = list(frame.columns)
            if list(frame.columns) != columns:
                raise ValueError(f'Schema mismatch in {filename}')
            label = identify_label(frame)
            if frame[label].isna().any():
                raise ValueError(f'Missing labels found in {filename}; inspect before sampling')
            frame['_source_file'] = filename
            frame['_source_row'] = np.arange(offset, offset + len(frame))
            frame['_priority'] = rng.random(len(frame))
            offset += len(frame)
            for name, group in frame.groupby(label, sort=True):
                counts[str(name)] += len(group)
                reservoirs[name] = retain_by_priority(reservoirs.get(name), group, per_class)
        file_counts[filename] = offset
        print(f'Scanned {filename}: {offset:,} rows', flush=True)
    frame = pd.concat([reservoirs[k] for k in sorted(reservoirs)], ignore_index=True)
    frame = frame.drop(columns='_priority')
    provenance = {'metadata': metadata, 'seed': seed, 'per_class': per_class,
                  'batch_size': batch_size, 'source_class_counts': dict(counts),
                  'source_file_counts': file_counts, 'sample_records': len(frame),
                  'sampling': 'Uniform random priority reservoir per raw label across every CSV. Class-capped development sample; not natural traffic prevalence.'}
    return frame, provenance
