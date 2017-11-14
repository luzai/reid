import numpy as np

from .features import extract_features
from .metrics import pairwise_distance

global_margin = 0


def mine_hard_pairs(model, data_loader, margin=0):
    model.eval()
    # Compute pairwise distance
    features = extract_features(model, data_loader, print_freq=1)
    distmat = pairwise_distance(features)
    distmat = distmat.cpu().numpy()
    # Get the pids
    dataset = data_loader.dataset.dataset
    pids = np.asarray([pid for _, pid, _ in dataset])
    # Find the hard triplets
    pairs = []
    for i, d in enumerate(distmat):
        pos_indices = np.where(pids == pids[i])[0]
        threshold = max(d[pos_indices]) + margin
        neg_indices = np.where(pids != pids[i])[0]
        pairs_g = [(i, p) for p in pos_indices]
        pairs.extend(pairs_g)
        pairs_n = [(i, n) for n in neg_indices if threshold >= d[n]]
        if len(pairs_n) < len(pairs_g):
            print('!!! waring pos  {} neg  {} '.format(len(pairs_g), len(pairs_n)))
        pairs.extend(pairs_n)
    return pairs


def mine_hard_triplets(model, data_loader, margin=0):
    model.eval()
    # Compute pairwise distance
    features = extract_features(model, data_loader, print_freq=1)
    distmat = pairwise_distance(features)
    distmat = distmat.cpu().numpy()
    # Get the pids
    dataset = data_loader.dataset.dataset
    pids = np.asarray([pid for _, pid, _ in dataset])
    # Find the hard triplets
    triplets = []
    for i, d in enumerate(distmat):
        pos_indices = np.where(pids == pids[i])[0]
        neg_indices = np.where(pids != pids[i])[0]
        sorted_pos = np.argsort(d[pos_indices])[::-1]
        for j in sorted_pos:
            p = pos_indices[j]
            mask = (d[neg_indices] <= d[p] + margin)
            neg_indices = neg_indices[mask]
            triplets.extend([(i, p, n) for n in neg_indices])
    return triplets
