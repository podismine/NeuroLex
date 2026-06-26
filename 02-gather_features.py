import argparse
from tqdm import tqdm
import os
import pickle
import pandas as pd
import numpy as np
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
    p = argparse.ArgumentParser(description="Compute temporal features from inferred token sequences.")
    p.add_argument("--token_folder", default="results/token_results", type=str, help="Directory containing one subdirectory per dataset/group.")
    p.add_argument("--save_file", default="results/temporal_features.csv", type=str, help="Output CSV path for temporal features.")
    p.add_argument("--token_num", default=12, type=int, help="Number of brain-state tokens.")
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
        ret[f"pro_{i}"] = (inputs['token'] == i).sum() / len(inputs['token'])
        ret[f"log_pro_{i}"] = logit(ret[f"pro_{i}"] + eps)
        ret[f"dwell_{i}"] = avg_dwell[i] * dowmsample_tr if i in avg_dwell else 0

    return ret

def map_sex(x):
    if x in ['F','female', 'f', 'Female']:
        return 1
    if x in ['m','male', 'M', 'Male']:
        return 0
    raise ValueError(f"Unsupported sex value: {x!r}. Use male/female (or M/F).")

def main():
    seed_everything()
    args = get_args()

    if not os.path.isdir(args.token_folder):
        raise FileNotFoundError(f"Token directory does not exist: {args.token_folder}")

    all_results = []
    groups = sorted(
        name for name in os.listdir(args.token_folder)
        if os.path.isdir(os.path.join(args.token_folder, name))
    )
    if not groups:
        raise ValueError(f"No group subdirectories found in {args.token_folder}")

    for folders in tqdm(groups):
        print(f"Processing folder: {folders}")
        folder_path = os.path.join(args.token_folder, folders)

        files = sorted(
            name for name in os.listdir(folder_path)
            if name.endswith(".pkl") and os.path.isfile(os.path.join(folder_path, name))
        )
        for file_case in files:
            file_path = os.path.join(folder_path, file_case)
            res = make_features(file_path, args.token_num)
            res['group'] = folders
            res['sex'] = map_sex(res['sex'])
            all_results.append(res)

    if not all_results:
        raise ValueError(f"No .pkl token files found in {args.token_folder}")
    output_dir = os.path.dirname(args.save_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(all_results).to_csv(args.save_file, index=False)


if __name__ == "__main__":
    main()
