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
    p.add_argument("--save_gamlss", default="results/gamlss_model", type=str, help="Path to the directory where GAMLSS models will be saved.")
    args = p.parse_args()
    return args


def main():
    seed_everything()
    args = get_args()
    df = pd.read_csv(args.input_file)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
    train_df = df[df["group"].isin(["train"])].copy().reset_index(drop=True)

    to_save_dir = args.save_gamlss
    os.makedirs(to_save_dir, exist_ok=True)

    run_feas = [c for c in df.columns if c.startswith("log_pro") or c.startswith("dwell")]
    run_feas = sorted(run_feas)

    print(f"Total features found: {len(run_feas)}")
    print("Feature list:", run_feas)

    for feature in run_feas:
        print(f"Running GAMLSS for feature: {feature}")
        model_path = f"{to_save_dir}/model_{feature}.rds"

        func_name = "SHASH"
        model = Gamlss(
            model_name=f"Gamlss_{feature}",
            x_vals=["age", "sex", "site"],
            y_val=feature,
        )

        print(f"[INFO] Running... {feature}")
        model.fit(
            r_code="""
                model <- gamlss(
                    y ~ pb(log(age), df=2, by = as.factor(sex)) +
                        as.factor(sex) +
                        re(random=~1|as.factor(site)),
                    sigma.formula = ~ log(age) + as.factor(sex),
                    nu.formula    = ~ 1,
                    tau.formula   = ~ 1,
                    family = SHASH(),
                    method = RS(500)
                )
            """,
            data=train_df,
        )
        model.save_model(model_path)
        print(f"[INFO] Done: {feature}")

if __name__ == "__main__":
    main()