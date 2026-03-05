import argparse
import torch
from tqdm import tqdm
import os
import pickle
import pandas as pd
import numpy as np
import random
from utils.utils import seed_everything
from collections import defaultdict

def logit(p, eps=1e-6):
    """Numerically stable logit transform."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def compute_dwell_times(sequence):
    sequence = np.array(sequence)
    dwell_times = []
    current_state = sequence[0]
    count = 1

    for i in range(1, len(sequence)):
        if sequence[i] == current_state:
            count += 1
        else:
            dwell_times.append((current_state, count))
            current_state = sequence[i]
            count = 1

    dwell_times.append((current_state, count))
    return dwell_times

def average_dwell_time(sequence):
    dwell = compute_dwell_times(sequence)
    dwell_by_state = defaultdict(list)
    
    for state, duration in dwell:
        dwell_by_state[state].append(duration)
    
    avg_dwell = {state: np.mean(times) for state, times in dwell_by_state.items()}
    return avg_dwell

def get_args():
    p = argparse.ArgumentParser(description="Infer tokens using the trained NeuroLex model.")
    p.add_argument("--token_folder", default="results/token_results", type=str, help="Path to the input data folder for inference.")
    p.add_argument("--save_file", default="results/temporal_features.csv", type=str, help="Path to the temporal features data frame.")
    p.add_argument("--token_num", default=12, type=int, help="Number of tokens to infer.")
    args = p.parse_args()
    return args

def make_features(file_path, token_num, eps = 1e-3, dowmsample_tr = 2.0):
    ret = dict()
    with open(file_path, 'rb') as ff:
        inputs = pickle.load(ff)

    for k in inputs.keys():
        if k == 'token' or k == 'timeseries': 
            continue
        ret[k] = inputs[k]
    
    avg_dwell = average_dwell_time(inputs['token'])
    for i in range(token_num):
        if f"pro_{i}" not in ret.keys(): ret[f"pro_{i}"] = []
        if f"log_pro_{i}" not in ret.keys(): ret[f"log_pro_{i}"] = []
        if f"dwell_{i}" not in ret.keys(): ret[f"dwell_{i}"] = []

        ret[f"pro_{i}"] = (inputs['token'] == i).sum() / len(inputs['token'])
        ret[f"log_pro_{i}"] = logit(ret[f"pro_{i}"] + eps)
        ret[f"dwell_{i}"] = avg_dwell[i] * dowmsample_tr if i in avg_dwell else 0

    return ret

def map_sex(x):
    if x in ['F','female', 'f', 'Female']:
        return 1
    if x in ['m','male', 'M', 'Male']:
        return 0

def main():
    seed_everything()
    args = get_args()

    for folders in tqdm(os.listdir(args.token_folder)):
        print(f"Processing folder: {folders}")
        folder_path = os.path.join(args.token_folder, folders)

        files = os.listdir(folder_path)
        all_results = []
        for file_case in files:
            file_path = os.path.join(folder_path, file_case)
            res = make_features(file_path, args.token_num)
            res['group'] = folders
            res['sex'] = map_sex(res['sex'])
            all_results.append(res)

        save_df = pd.DataFrame(all_results)
        save_df.to_csv(args.save_file, index=None)


if __name__ == "__main__":
    main()