import os
import argparse
import warnings
import pandas as pd
import numpy as np
from utils.utils import seed_everything
from gamlss_python.gamlss_main import Gamlss
from gamlss_python.site_transfer_utils import site_transfer
warnings.filterwarnings("ignore")

def get_args():
    p = argparse.ArgumentParser(description="Transfer GAMLSS normative models to a new site or dataset.")
    p.add_argument("--input_file", default="results/temporal_features.csv", type=str, help="Path to the input data frame with temporal features.")
    p.add_argument("--new_data", default="results/new_site_temporal_features.csv", type=str, help="Path to the new input data.")
    p.add_argument("--gamlss_model", default="results/gamlss_model", type=str, help="Directory containing fitted GAMLSS models.")
    p.add_argument("--proxy_site", required=True, type=str, help="Reference site in the training data used for transfer.")
    p.add_argument("--fraction", default=1.0, type=float, help="Fraction of eligible new-site data used for transfer fitting.")
    p.add_argument("--output_file", default="results/new_site_temporal_features_with_zscore.csv", type=str, help="Output CSV path for transferred z-scores.")
    p.add_argument("--save_transfer_models", default=None, type=str, help="Optional directory in which to save site-specific adjustment models.")
    args = p.parse_args()
    return args

def main():
    seed_everything()
    args = get_args()
    df = pd.read_csv(args.input_file)
    train_df = df[df["group"].isin(["train"])].reset_index(drop=True)
    new_site_df = pd.read_csv(args.new_data)
    if train_df.empty:
        raise ValueError("No training rows found. Set the training folder name to 'train'.")
    if "site" not in new_site_df:
        raise ValueError("The new-site CSV must contain a 'site' column.")

    run_feas = [c for c in df.columns if c.startswith("log_pro") or c.startswith("dwell")]
    run_feas = sorted(run_feas)
    missing_features = set(run_feas) - set(new_site_df.columns)
    if missing_features:
        raise ValueError(f"New-site data is missing features: {sorted(missing_features)}")

    print(f"Total features found: {len(run_feas)}")
    print("Feature list:", run_feas)
    for feature in run_feas:
        print(f"Running GAMLSS Transfer for feature: {feature}")
        model_path = f"{args.gamlss_model}/model_{feature}.rds"
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing model for {feature}: {model_path}")

        print(f"[INFO] Running... {feature}")
        sites = new_site_df["site"].unique()
        for site in sites:
            site_mask = new_site_df["site"] == site
            save_folder = None
            if args.save_transfer_models:
                save_folder = os.path.join(args.save_transfer_models, feature)
            predictions_df, adjusted_predictions, z_scores, train_set_index, new_model = site_transfer(
                model_path,
                train_df,
                new_site_df,
                site,
                args.proxy_site,
                args.fraction,
                feature,
                ["age", "sex", "site"],
                500,
                save_folder=save_folder,
            )
            new_site_df.loc[site_mask, f"{feature}_z_score"] = z_scores

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    new_site_df.to_csv(args.output_file, index=False)
    print(f"[INFO] Saved transferred z-scores: {args.output_file}")

if __name__ == "__main__":
    main()
