"""Inspect metadata before streaming a bounded sample; never prepare the full dataset."""
import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "muzom/CIC-IDS2017"
CACHE = ROOT / "data" / "raw" / "hf_cache"


def inspect_dataset(revision=None):
    """Fetch repository and builder metadata without downloading CSV contents."""
    try:
        repo = HfApi().dataset_info(DATASET_ID, revision=revision, files_metadata=True)
        builder = load_dataset_builder(DATASET_ID, revision=repo.sha, cache_dir=str(CACHE))
        info = builder.info
        splits = {k: {"records": v.num_examples, "bytes": v.num_bytes}
                  for k, v in (info.splits or {}).items()}
        if not splits:
            # CSV builders may expose no counts/schema until preparation. Creating
            # streaming split objects discovers names without iterating CSV rows.
            streams = load_dataset(DATASET_ID, name=builder.config.name,
                                   revision=repo.sha, streaming=True, cache_dir=str(CACHE))
            splits = {k: {"records": None, "bytes": None} for k in streams}
        return {
            "dataset": DATASET_ID, "revision": repo.sha,
            "files": [{"path": f.rfilename, "bytes": f.size} for f in repo.siblings],
            "config": builder.config.name,
            "splits": splits,
            "features": info.features.to_dict() if info.features else {},
        }
    except Exception as exc:
        raise RuntimeError("Cannot inspect Hugging Face dataset. Check internet access and "
                           "dependencies; retry: python -m src.data_loader --inspect-only") from exc


def load_sample(records=10000, seed=42, buffer_size=10000, split="train", metadata=None):
    """Buffered shuffle is reproducible at a pinned revision, not a uniform global sample.

    Memory scales with records + buffer_size. Streaming may prefetch additional bytes.
    File/time ordering can exclude rare attacks, even with a shuffled buffer.
    """
    if records < 1 or buffer_size < 1:
        raise ValueError("records and buffer_size must be positive")
    metadata = metadata if metadata is not None else inspect_dataset()
    if metadata["splits"] and split not in metadata["splits"]:
        raise ValueError(f"Unknown split {split!r}; available: {list(metadata['splits'])}")
    try:
        stream = load_dataset(DATASET_ID, name=metadata["config"],
                              revision=metadata["revision"], split=split,
                              streaming=True, cache_dir=str(CACHE))
        frame = pd.DataFrame(list(stream.shuffle(seed=seed, buffer_size=buffer_size).take(records)))
        if frame.empty:
            raise ValueError("The selected stream returned no records")
        return frame
    except Exception as exc:
        raise RuntimeError("Cannot stream dataset records. Check network access and retry "
                           "python -m src.data_loader --records 1000") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--revision", default=None, help="Pin a Hugging Face commit for reproducibility")
    args = parser.parse_args()
    try:
        metadata = inspect_dataset(revision=args.revision)
        print(json.dumps(metadata, indent=2))
        if not args.inspect_only:
            from src.quality import write_report
            frame = load_sample(args.records, args.seed, args.buffer_size, args.split, metadata)
            print(frame.dtypes.to_string())
            print(frame.head().to_string())
            path = write_report(frame, metadata, args.seed, args.buffer_size, args.split)
            print(f"Report saved to {path}")
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\nCause: {exc.__cause__ or exc}\n")


if __name__ == "__main__":
    main()
