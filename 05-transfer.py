import os
import argparse
import warnings
import pandas as pd
import numpy as np
from utils.utils import seed_everything
from rpy2 import robjects as ro
from gamlss_python.gamlss_main import Gamlss
from gamlss_python.site_transfer_utils import site_transfer
warnings.filterwarnings("ignore")

def get_args():
    p = argparse.ArgumentParser(description="Infer tokens using the trained NeuroLex model.")
    p.add_argument("--input_file", default="results/temporal_features.csv", type=str, help="Path to the input data frame with temporal features.")
    p.add_argument("--new_data", default="results/new_site_temporal_features.csv", type=str, help="Path to the new input data.")
    p.add_argument("--gamlss_model", default="results/gamlss_model", type=str, help="Path to the directory where GAMLSS models will be saved.")
    args = p.parse_args()
    return args

def main():
    seed_everything()
    args = get_args()
    df = pd.read_csv(args.input_file)
    train_df = df[df["group"].isin(["train"])].reset_index(drop=True)
    new_site_df = pd.read_csv(args.new_data)

    to_save_dir = args.save_gamlss
    os.makedirs(to_save_dir, exist_ok=True)

    run_feas = [c for c in df.columns if c.startswith("log_pro") or c.startswith("dwell")]
    run_feas = sorted(run_feas)

    print(f"Total features found: {len(run_feas)}")
    print("Feature list:", run_feas)
    func_name = "SHASH"

    for feature in run_feas:
        print(f"Running GAMLSS Transfer for feature: {feature}")
        model_path = f"{to_save_dir}/model_{feature}.rds"

        print(f"[INFO] Running... {feature}")
        sites = new_site_df["site"].unique()
        proxy = 'HBN-Site-CBIC-Siemens_Prisma_fit'
        frac = 1.0

        for site in sites:
            predictions_df, adjusted_predictions, z_scores, train_set_index, new_model = site_transfer(
                model_path,
                train_df,
                new_site_df,
                site,
                proxy,
                frac,
                feature,
                ["age", "sex", "site"],
                500,
            )

if __name__ == "__main__":
    main()