import os
import argparse
import warnings
import pandas as pd
import numpy as np
from utils.utils import seed_everything
from rpy2 import robjects as ro
from gamlss_python.gamlss_main import Gamlss
warnings.filterwarnings("ignore")

def get_args():
    p = argparse.ArgumentParser(description="Infer tokens using the trained NeuroLex model.")
    p.add_argument("--input_file", default="results/temporal_features.csv", type=str, help="Path to the input data frame with temporal features.")
    p.add_argument("--load_model", default="results/gamlss_model", type=str, help="Path to the directory of the saved GAMLSS model.")
    p.add_argument("--output_file", default="results/temporal_features_with_zscore.csv", type=str, help="Path to the output file with z-scores.")
    args = p.parse_args()
    return args

def main():
    seed_everything()
    args = get_args()
    df = pd.read_csv(args.input_file)
    train_df = df[df["group"].isin(["train"])].reset_index(drop=True) # my case is "df_group"

    run_feas = [c for c in df.columns if c.startswith("log_pro") or c.startswith("dwell")]
    run_feas = sorted(run_feas)

    for feature in run_feas:
        print(f"[INFO] Processing feature: {feature}")
        model_path = f"{args.load_model}/model_{feature}.rds"

        model = Gamlss.load_model(model_path)
        print(f"[INFO] Done loading model: {model_path}")
        z_score = model.z_score(df, train_df)
    df[f"{feature}_z_score"] = z_score
    df.to_csv(args.output_file, index=None)

if __name__ == "__main__":
    main()