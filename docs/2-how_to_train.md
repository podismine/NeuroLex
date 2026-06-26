# Fit a normative GAMLSS model

This workflow fits new feature-specific GAMLSS models. It requires a sufficiently large, quality-controlled normative cohort and should not be run on the demo data.

## Prepare the feature table

First create token files and temporal features as described in [the inference guide](1-how_to_inference.md). Put the normative cohort token files in `results/token_results/train/`; this ensures their rows have `group == "train"`.

The feature CSV must contain `age`, `sex`, `site`, `group`, and the `log_pro_*`/`dwell_*` features. `sex` is encoded as 0 (male) or 1 (female) by `02-gather_features.py`.

## Fit models

```bash
python 03-fit_gamlss.py \
  --input_file results/temporal_features.csv \
  --save_gamlss results/gamlss_model
```

One RDS model is created for each logit fractional-occupancy and dwell-time feature. The model fits age and sex effects and includes a site random effect.

## Calculate z-scores

```bash
python 04-infer_zscore.py \
  --input_file results/temporal_features.csv \
  --load_model results/gamlss_model \
  --output_file results/temporal_features_with_zscore.csv
```

The output adds one `<feature>_z_score` column for every fitted feature.

## Practical considerations

- Keep acquisition, preprocessing, network representation, and metadata coding consistent across cohorts.
- Check age coverage, sex balance, and site sample sizes before fitting.
- Use `05-transfer.py` to calibrate a model for a new site rather than silently pooling incompatible datasets.
