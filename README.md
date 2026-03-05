# 🧠 A normative reference for large-scale human brain dynamics across the lifespan

🔗 [https://www.biorxiv.org/content/10.64898/2026.03.06.710057v1.abstract](https://www.biorxiv.org/content/10.64898/2026.03.06.710057v1.abstract)

Official repository for *“A normative reference for large-scale human brain dynamics across the lifespan”*

---

## 🚀 Overview

This repository provides the core pipeline for constructing a normative reference of large-scale brain dynamics based on **Brain State Tokens (BSTs)** and **temporal features**.

---

## 🧩 Pipeline

The full workflow is illustrated below:

```bash
python 01-infer_tokens.py \
    --model_path model/model.pth \
    --input_data demo_data/anonymous \
    --output_path results/token_results/train
# Infer brain state tokens

python 02-gather_features.py \
    --token_folder results/token_results \
    --save_file results/temporal_features.csv
# Compute temporal measurements (Fractional Occupancy, Dwell Time)

python 03-fit_gamlss.py \
    --input_file results/temporal_features.csv \
    --save_gamlss results/gamlss_model
# Fit GAMLSS model for lifespan normative modeling

python 04-infer_zscore.py \
    --input_file results/temporal_features.csv \
    --load_model results/gamlss_model \
    --output_file results/temporal_features_with_zscore.csv
# Infer normative z-scores

python 05-transfer.py \
    --input_file results/temporal_features.csv \
    --new_data results/new_site_temporal_features.csv \
    --gamlss_model results/gamlss_model
# Transfer model to a new site or dataset
```

---

## 📂 Repository Structure

```bash
demo_data/        # Example data for demonstration purposes
                 # Note: limited sample size, not sufficient for GAMLSS fitting
                 # TODO: provide more anonymized data compliant with sharing policies

model/            # Model checkpoints
                 # Includes NeuroLex model and (planned) GAMLSS models
                 # TODO: upload GAMLSS model
```

---

## ⚙️ Dependencies

* Python packages: see environment.yml

This project additionally relies on a Python–R hybrid normative modeling framework:

* **GAMLSS (Python–R interface)**
  🔗 [https://github.com/nsharma3150/GAMLSS-python](https://github.com/nsharma3150/GAMLSS-python)

> ⚠️ The package is currently private and will be released soon.
> For early access, please contact [Us](https://mhm-lab.github.io/).

---

## 📌 Notes

* Additional non-core utilities will be uploaded progressively.
* We are developing a self-contained implementation of the GAMLSS-based normative models, where all necessary distribution parameters are encapsulated within the trained model.
* **More documents are detailed in docs: [1.inference](https://github.com/podismine/NeuroLex/blob/main/docs/1-how_to_inference.md); [2.train](https://github.com/podismine/NeuroLex/blob/main/docs/2-how_to_train.md)**
