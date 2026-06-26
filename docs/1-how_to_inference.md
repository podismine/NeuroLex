# Run inference on your own data

Install the environment and the temporary GAMLSS dependency as described in the [README](../README.md#installation).

## 1. Prepare input files

Put one participant dictionary in each `.pkl` file. The directory name becomes the dataset/group label during feature extraction.

```python
import os
import pickle
import numpy as np

output_dir = "data/clinical"
os.makedirs(output_dir, exist_ok=True)

subject = {
    "timeseries": np.load("path/to/timeseries.npy"),  # (timepoints, 7)
    "age": 30,
    "sex": "female",
    "site": "my_site",
}
with open(os.path.join(output_dir, "participant_001.pkl"), "wb") as file:
    pickle.dump(subject, file)
```

`timeseries` must contain the same Yeo-7 representation and compatible preprocessing as the pretrained NeuroLex checkpoint.

## 2. Infer brain-state tokens

```bash
python 01-infer_tokens.py \
  --model_path model/model.pth \
  --input_data data/clinical \
  --output_path results/token_results/clinical
```

## 3. Extract temporal features

```bash
python 02-gather_features.py \
  --token_folder results/token_results \
  --save_file results/clinical_temporal_features.csv
```

The CSV includes `pro_*` (fractional occupancy), `log_pro_*` (logit-transformed occupancy), and `dwell_*` (mean dwell time in seconds).

## 4. Score the data

The supplied normative models are in `model/gamlss_model`. Normative z-score inference expects the feature CSV to include reference rows whose group is named `train`. If your input only contains a new dataset, first use the site-transfer workflow with an appropriate reference dataset; do not treat the small demo set as a normative reference.

```bash
python 04-infer_zscore.py \
  --input_file results/temporal_features.csv \
  --load_model model/gamlss_model \
  --output_file results/temporal_features_with_zscore.csv
```

For site transfer, provide the full reference feature CSV, new-site feature CSV, model directory, and a proxy site present in the reference data:

```bash
python 05-transfer.py \
  --input_file results/reference_temporal_features.csv \
  --new_data results/new_site_temporal_features.csv \
  --gamlss_model model/gamlss_model \
  --proxy_site YOUR_REFERENCE_SITE \
  --output_file results/new_site_temporal_features_with_zscore.csv
```
