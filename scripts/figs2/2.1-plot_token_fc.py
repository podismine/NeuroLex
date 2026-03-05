import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
from model.model import NeuroLex_model
import os, random
from utils.utils import seed_everything
map_name ={
    0:2,
    1:1,
    2:9,
    3:10,
    4:8,
    5:7,
    6:6,
    7:3,
    8:5,
    9:11,
    10:0,
    11:4,
}
seed_everything()

TOKEN_NUM = 12
checkpoints = f'model/model.pth'
model = NeuroLex_model(input_dim=7, condition_dim=1, hidden_dim=16,num_embeddings=TOKEN_NUM,latent_dim=8,commitment_cost=0.1,depth=4)
checkpoint = torch.load(checkpoints,map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

x = np.array([[i for i in range(TOKEN_NUM)]])
x_torch = torch.from_numpy(x).long()
codebook = model.vector_quantizer.embedding.weight
quantized = codebook[x_torch[0]]  # shape: [B, T, latent_dim]
x_hat = model.decoder(quantized).detach().numpy()

for i in range(7):
    for j in range(7):
        if i == j:
            x_hat[:,i,j] = 1

np.set_printoptions(precision=2, suppress=True)
dfc = x_hat

vmin, vmax = -1, 1
for idx in range(len(dfc)):
    fig, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(dfc[idx], vmin=vmin, vmax=vmax, cmap='RdBu_r')
    ax.set_title(f'BST {map_name[idx]+1}', fontsize=20)
    ax.set_xticks([])

    ax.set_yticks(range(7))
    ax.set_yticklabels(['VIS', 'SMN', 'DAN', 'VAN', 'LN', 'FPN', 'DMN'], fontsize=16)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f'results/figs/figs2/dfc_token_{map_name[idx]+1}.pdf',bbox_inches='tight')
    plt.close()
