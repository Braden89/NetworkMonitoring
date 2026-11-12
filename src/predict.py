"""Offline inference for compatible raw flow-feature tables (not packet capture)."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import ROOT
from src.preprocessing import normalize_headers


def predict_frame(frame, artifact):
    frame = normalize_headers(frame)
    required = artifact['raw_feature_columns']
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required raw features: {missing}')
    features = frame.loc[:, required]
    pipeline = artifact['pipeline']
    ids = pipeline.predict(features)
    return pd.DataFrame({'prediction': [artifact['label_mapping'][int(i)] for i in ids],
                         'attack_score': pipeline.predict_proba(features)[:, 1]}, index=frame.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='CSV or Parquet raw flow table')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/binary_detector.joblib')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError('Prediction output must differ from the input file')
        frame = pd.read_parquet(args.input) if args.input.suffix == '.parquet' else pd.read_csv(args.input)
        result = predict_frame(frame, joblib.load(args.model))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f'Saved {len(result):,} predictions to {args.output}')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Prediction failed: {exc}\n')


if __name__ == '__main__':
    main()
