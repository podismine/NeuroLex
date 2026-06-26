# NeuroLex

Official implementation of *[A normative reference for large-scale human brain dynamics across the lifespan](https://www.biorxiv.org/content/10.64898/2026.03.06.710057v1.abstract)*.

NeuroLex assigns fMRI time points to 12 brain-state tokens, derives temporal features (fractional occupancy and dwell time), and uses GAMLSS normative models to quantify individual deviations across the lifespan.

## What is included

- A pretrained 12-token NeuroLex checkpoint (`model/model.pth`)
- Example anonymized fMRI time series (`demo_data/anonymous/`)
- Pretrained GAMLSS models (`model/gamlss_model/`) for normative-score inference
- Scripts for token inference, feature extraction, GAMLSS fitting, z-score inference, and site transfer

The demo data are intentionally small and are suitable for testing the token/feature pipeline only; they are not sufficient for fitting a stable normative model.

## Installation

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate NeuroLex
```

NeuroLex also needs R and the R packages used by the GAMLSS backend:

```bash
R -e 'install.packages(c("gamlss", "gamlss.dist"), repos="https://cloud.r-project.org")'
```

### Temporary GAMLSS dependency

Until the dependency is updated at [nsharma3150/GAMLSS-python](https://github.com/nsharma3150/GAMLSS-python), install the bundled archive after creating the environment:

```bash
pip install ./model/GAMLSS-python-main.zip
```

The archive must be a complete ZIP file. If `pip` reports an archive or end-of-file error, copy/upload the archive again before retrying. Once the GitHub repository is updated, replace the command above with:

```bash
pip install "gamlss-python @ git+https://github.com/nsharma3150/GAMLSS-python.git"
```

## Quick start

Run the following commands from the repository root:

```bash
# 1. Infer a token sequence for each .pkl subject file.
python 01-infer_tokens.py \
  --input_data demo_data/anonymous \
  --output_path results/token_results/train

# 2. Turn token sequences into temporal features.
python 02-gather_features.py \
  --token_folder results/token_results \
  --save_file results/temporal_features.csv

# 3. Apply the bundled normative models.
python 04-infer_zscore.py \
  --input_file results/temporal_features.csv \
  --load_model model/gamlss_model \
  --output_file results/temporal_features_with_zscore.csv
```

`04-infer_zscore.py` requires a group named `train` to provide the normative reference. For an independent dataset, use the included model with the same feature definitions and provide the appropriate reference/transfer data; see [inference instructions](docs/1-how_to_inference.md).

## Input format

Each subject is stored as one `.pkl` file containing a dictionary:

```python
{
    "timeseries": np.ndarray,  # shape: (timepoints, 7), Yeo-7 parcel/network signals
    "age": 25,
    "sex": "female",          # female/F or male/M
    "site": "site_name"
}
```

Place files in a group directory such as `data/train/` or `data/clinical/`. Token inference preserves the metadata; feature extraction names the group from this directory.

## Workflow

| Step | Script | Output |
| --- | --- | --- |
| Infer tokens | `01-infer_tokens.py` | one token sequence per participant |
| Extract features | `02-gather_features.py` | a temporal-feature CSV |
| Fit models | `03-fit_gamlss.py` | one `.rds` model per feature |
| Score deviations | `04-infer_zscore.py` | feature-wise normative z-scores |
| Transfer sites | `05-transfer.py` | transferred model output from the GAMLSS backend |

See [how to run inference](docs/1-how_to_inference.md) and [how to fit a normative model](docs/2-how_to_train.md) for full commands and requirements.

## Notes

- Use consistent preprocessing and the same 7-network representation as the pretrained model.
- The packaged GAMLSS models have been sanitized to remove participant-level training data.
- Site transfer requires a reference site present in the training data and enough data in the new site for calibration.

## Citation

If you use NeuroLex, please cite the associated preprint above. Citation details will be updated with the peer-reviewed publication.

## License

This project is distributed under the [MIT License](LICENSE).
