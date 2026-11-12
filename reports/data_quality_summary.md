# Data quality summary

Generated (UTC): 2026-11-12T04:07:46.159283+00:00

## Scope

Source: https://huggingface.co/datasets/muzom/CIC-IDS2017
Revision: `b33f3052398ffd63463ce3120fec2acbb68e2ff6`; split: `train`; seed: 42; shuffle buffer: 10000.
All quality statistics below describe only the examined sample. Buffered streaming is not a uniform global sample; file/time ordering can omit attack categories. No full-dataset quality claims are made.

Records examined: 10000; feature columns (excluding label): 78.
Label column: ` Label`; duplicate rows beyond first occurrence: 116.

## Observed attack categories and class distribution

| Category | Count | Sample % |
|---|---:|---:|
| BENIGN | 9997 | 99.970 |
| PortScan | 3 | 0.030 |

## Missing values

Total: 5

```json
{
  "Flow Bytes/s": 5
}
```

## Infinite numerical values

Total: 31

```json
{
  "Flow Bytes/s": 13,
  " Flow Packets/s": 18
}
```

## Nonnumeric values in non-label text columns

Total: 0

```json
{}
```

## Negative numerical values (review domain meaning)

Total: 9523

```json
{
  " Flow IAT Min": 26,
  " Fwd Header Length": 16,
  " Bwd Header Length": 13,
  " Fwd Header Length.1": 16,
  "Init_Win_bytes_forward": 4075,
  " Init_Win_bytes_backward": 5361,
  " min_seg_size_forward": 16
}
```

## Potential preprocessing problems

Constant columns in sample: [' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' CWE Flag Count', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
Column names with outer whitespace: [' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', ' Active Std', ' Active Max', ' Active Min', ' Idle Std', ' Idle Max', ' Idle Min', ' Label']
Review missing/infinite values, numeric parsing, negative sentinel values, class imbalance, duplicated flows, and redundant features before training. Constant columns may vary outside this sample. Duplicate counting uses all columns including the label; conflicting labels require separate investigation. Preserve temporal/file provenance when designing future splits to prevent leakage. No rows or columns were cleaned here.

## Observed sample columns and types

```json
{
  " Destination Port": "int64",
  " Flow Duration": "int64",
  " Total Fwd Packets": "int64",
  " Total Backward Packets": "int64",
  "Total Length of Fwd Packets": "int64",
  " Total Length of Bwd Packets": "int64",
  " Fwd Packet Length Max": "int64",
  " Fwd Packet Length Min": "int64",
  " Fwd Packet Length Mean": "float64",
  " Fwd Packet Length Std": "float64",
  "Bwd Packet Length Max": "int64",
  " Bwd Packet Length Min": "int64",
  " Bwd Packet Length Mean": "float64",
  " Bwd Packet Length Std": "float64",
  "Flow Bytes/s": "float64",
  " Flow Packets/s": "float64",
  " Flow IAT Mean": "float64",
  " Flow IAT Std": "float64",
  " Flow IAT Max": "int64",
  " Flow IAT Min": "int64",
  "Fwd IAT Total": "int64",
  " Fwd IAT Mean": "float64",
  " Fwd IAT Std": "float64",
  " Fwd IAT Max": "int64",
  " Fwd IAT Min": "int64",
  "Bwd IAT Total": "int64",
  " Bwd IAT Mean": "float64",
  " Bwd IAT Std": "float64",
  " Bwd IAT Max": "int64",
  " Bwd IAT Min": "int64",
  "Fwd PSH Flags": "int64",
  " Bwd PSH Flags": "int64",
  " Fwd URG Flags": "int64",
  " Bwd URG Flags": "int64",
  " Fwd Header Length": "int64",
  " Bwd Header Length": "int64",
  "Fwd Packets/s": "float64",
  " Bwd Packets/s": "float64",
  " Min Packet Length": "int64",
  " Max Packet Length": "int64",
  " Packet Length Mean": "float64",
  " Packet Length Std": "float64",
  " Packet Length Variance": "float64",
  "FIN Flag Count": "int64",
  " SYN Flag Count": "int64",
  " RST Flag Count": "int64",
  " PSH Flag Count": "int64",
  " ACK Flag Count": "int64",
  " URG Flag Count": "int64",
  " CWE Flag Count": "int64",
  " ECE Flag Count": "int64",
  " Down/Up Ratio": "int64",
  " Average Packet Size": "float64",
  " Avg Fwd Segment Size": "float64",
  " Avg Bwd Segment Size": "float64",
  " Fwd Header Length.1": "int64",
  "Fwd Avg Bytes/Bulk": "int64",
  " Fwd Avg Packets/Bulk": "int64",
  " Fwd Avg Bulk Rate": "int64",
  " Bwd Avg Bytes/Bulk": "int64",
  " Bwd Avg Packets/Bulk": "int64",
  "Bwd Avg Bulk Rate": "int64",
  "Subflow Fwd Packets": "int64",
  " Subflow Fwd Bytes": "int64",
  " Subflow Bwd Packets": "int64",
  " Subflow Bwd Bytes": "int64",
  "Init_Win_bytes_forward": "int64",
  " Init_Win_bytes_backward": "int64",
  " act_data_pkt_fwd": "int64",
  " min_seg_size_forward": "int64",
  "Active Mean": "float64",
  " Active Std": "float64",
  " Active Max": "int64",
  " Active Min": "int64",
  "Idle Mean": "float64",
  " Idle Std": "float64",
  " Idle Max": "int64",
  " Idle Min": "int64",
  " Label": "str"
}
```

## Repository metadata (not computed sample statistics)

Null counts and empty feature metadata mean the source builder did not supply them; sample types are listed above.

```json
{
  "dataset": "muzom/CIC-IDS2017",
  "revision": "b33f3052398ffd63463ce3120fec2acbb68e2ff6",
  "files": [
    {
      "path": ".gitattributes",
      "bytes": 2960
    },
    {
      "path": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
      "bytes": 77123859
    },
    {
      "path": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
      "bytes": 76906168
    },
    {
      "path": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
      "bytes": 58316725
    },
    {
      "path": "LICENSE",
      "bytes": 706
    },
    {
      "path": "Monday-WorkingHours.pcap_ISCX.csv",
      "bytes": 176927918
    },
    {
      "path": "README.md",
      "bytes": 771
    },
    {
      "path": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
      "bytes": 83102436
    },
    {
      "path": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
      "bytes": 52023263
    },
    {
      "path": "Tuesday-WorkingHours.pcap_ISCX.csv",
      "bytes": 135078995
    },
    {
      "path": "Wednesday-workingHours.pcap_ISCX.csv",
      "bytes": 225166395
    }
  ],
  "config": "default",
  "splits": {
    "train": {
      "records": null,
      "bytes": null
    }
  },
  "features": {}
}
```
