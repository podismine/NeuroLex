from model.model import NeuroLex_model
import argparse
import torch
from tqdm import tqdm
import os
import pickle
import numpy as np
import random
from scipy.signal import hilbert, detrend
from utils.utils import seed_everything


def get_args():
    p = argparse.ArgumentParser(description="Infer tokens using the trained NeuroLex model.")
    p.add_argument("--model_path", default="model/model.pth", type=str, help="Path to the trained model checkpoint.")
    p.add_argument("--input_data", default="demo_data/anonymous", type=str, help="Path to the input data folder for inference.")
    p.add_argument("--output_path", default="results/token_results/train", type=str, help="Path to save the inferred tokens.")
    args = p.parse_args()
    return args

def dFC_fast(bold):
    bold = bold.T
    bold = detrend(bold, axis=1)
    bold -= bold.mean(axis=1, keepdims=True)
    analytic_signal = hilbert(bold, axis=1)
    phase = np.angle(analytic_signal)

    complex_phase = np.exp(1j * phase)

    dFC = np.real(np.einsum('it,jt->ijt', complex_phase, np.conj(complex_phase)))
    return dFC.transpose(2,0,1)

def main():
    seed_everything()
    args = get_args()
    files = os.listdir(args.input_data)
    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuroLex_model(input_dim=7, condition_dim=1, hidden_dim=16,num_embeddings=12,latent_dim=8,commitment_cost=0.1,depth=4)
    model.load_state_dict(torch.load(args.model_path,map_location='cpu'))
    model.to(device)

    for file_case in tqdm(files):
        file_path = os.path.join(args.input_data, file_case)
        save_path = os.path.join(args.output_path, file_case)
        cache_dict = dict()

        with open(file_path, 'rb') as ff:
            inputs = pickle.load(ff)

        phase_lock_matrix = dFC_fast(inputs['timeseries'])
        input_tensor = torch.Tensor(phase_lock_matrix).to(device)  # Add batch dimension

        with torch.no_grad():
            tokens = model.infer_token(input_tensor).cpu().numpy().reshape(-1)

        cache_dict['token'] = tokens
        for key in inputs.keys():
            cache_dict[key] = inputs[key]
        cache_dict['file'] = file_case
        with open(save_path, 'wb') as ff:
            pickle.dump(cache_dict, ff)

if __name__ == "__main__":
    main()