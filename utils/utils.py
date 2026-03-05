import os
import random
from typing import List, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Adapted from CEBRA:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Schneider, S., Lee, J. H., and Mathis, M. W. (2023).
# Learnable latent embeddings for joint behavioural and neural analysis.
# Nature, 617, 360-368. https://doi.org/10.1038/s41586-023-06031-6
def _consistency_scores_numpy2(
    embeddings: List[np.ndarray],
    datasets: List[Union[int, str]],
) -> Tuple[List[float], List[tuple]]:
    """
    Compute consistency scores (R²) using NumPy-based linear regression
    with Hungarian matching for token alignment.

    Parameters
    ----------
    embeddings : list of np.ndarray
        A list of codebook matrices with shape (K, D).
    datasets : list of int or str
        Dataset identifiers corresponding to each embedding.

    Returns
    -------
    scores : list of float
        R² scores for each ordered embedding pair.
    pairs : list of tuple
        Dataset ID pairs in the form (dataset_i, dataset_j).
    """
    if len(embeddings) <= 1:
        raise ValueError("At least two embeddings are required for comparison.")
    if datasets is None or len(datasets) != len(embeddings):
        raise ValueError(
            "Dataset identifiers must be provided and match the number of embeddings."
        )

    scores = []
    pairs = []

    for n, embedding_a in enumerate(embeddings):
        for m, embedding_b in enumerate(embeddings):
            if n == m:
                continue

            # Hungarian matching based on Euclidean distance
            cost = np.linalg.norm(
                embedding_a[:, None, :] - embedding_b[None, :, :],
                axis=-1,
            )
            row_ind, col_ind = linear_sum_assignment(cost)

            embedding_a_matched = embedding_a[row_ind]
            embedding_b_matched = embedding_b[col_ind]

            # Linear regression: B = A @ W + b
            x = np.hstack(
                [embedding_a_matched, np.ones((embedding_a_matched.shape[0], 1))]
            )
            y = embedding_b_matched

            w = np.linalg.pinv(x.T @ x) @ x.T @ y
            y_pred = x @ w

            # R² score
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
            r2 = 1 - ss_res / ss_tot

            scores.append(r2)
            pairs.append((datasets[n], datasets[m]))

    return scores, pairs

def calc_half_split_pcc(label, index, K: int = 12) -> float:
    """
    Compute half-split reliability (Pearson correlation) of token fractional occupancy.

    Parameters
    ----------
    label : np.ndarray
        Token sequence of shape (T,), e.g., [1, 2, 3, 1, 2, 3].
    index : np.ndarray
        Subject indices of shape (T,), e.g., [0, 0, 0, 1, 1, 1].
    K : int, default=12
        Number of discrete token states.

    Returns
    -------
    float
        Global Pearson correlation coefficient between the two halves.
    """
    from scipy.stats import pearsonr
    import numpy as np

    label = np.asarray(label).reshape(-1)
    index = np.asarray(index).reshape(-1)

    subjects = np.unique(index)

    fo1_list = []
    fo2_list = []

    for sid in subjects:
        mask = index == sid
        tokens = label[mask]

        # Skip subjects with insufficient length
        if len(tokens) < 2:
            continue

        half_len = len(tokens) // 2
        s1 = tokens[:half_len]
        s2 = tokens[-half_len:]

        # Fractional occupancy
        fo1 = np.bincount(s1, minlength=K) / len(s1)
        fo2 = np.bincount(s2, minlength=K) / len(s2)

        fo1_list.append(fo1)
        fo2_list.append(fo2)

    if len(fo1_list) == 0:
        raise ValueError("No valid subjects with sufficient sequence length.")

    fo1_mat = np.vstack(fo1_list)
    fo2_mat = np.vstack(fo2_list)

    r_global, _ = pearsonr(fo1_mat.flatten(), fo2_mat.flatten())

    return float(r_global)

def calc_half_split_icc(label, K: int = 12) -> float:
    """
    Compute half-split reliability (ICC3) of token fractional occupancy.

    Parameters
    ----------
    label : np.ndarray
        Token sequence of shape (T,).
    K : int, default=12
        Number of discrete token states.

    Returns
    -------
    float
        ICC3 value between the two halves.
    """
    import numpy as np
    import pandas as pd
    import pingouin as pg

    label = np.asarray(label).reshape(-1)

    tokens = label.astype(int)
    half_len = len(tokens) // 2

    if half_len == 0:
        raise ValueError("Token sequence is too short for half-split ICC computation.")

    split1 = tokens[:half_len]
    split2 = tokens[-half_len:]

    fo1 = np.bincount(split1, minlength=K) / len(split1)
    fo2 = np.bincount(split2, minlength=K) / len(split2)

    fo1_flat = fo1.flatten()
    fo2_flat = fo2.flatten()
    n = fo1_flat.shape[0]

    df = pd.DataFrame(
        {
            "target": np.arange(n),
            "rater1": fo1_flat,
            "rater2": fo2_flat,
        }
    )

    df_long = df.melt(
        id_vars="target",
        value_vars=["rater1", "rater2"],
        var_name="rater",
        value_name="score",
    )

    icc_table = pg.intraclass_corr(
        data=df_long,
        targets="target",
        raters="rater",
        ratings="score",
    )

    icc3 = icc_table.loc[icc_table["Type"] == "ICC3", "ICC"].values[0]

    return float(icc3)