import numpy as np
from lz import *
from reid.evaluators import extract_features, pairwise_distance
from torch.utils.data import DataLoader


def mine_hard_pairs(model, data_loader, margin=0.32):
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
        pairs.extend([(i, p) for p in pos_indices])
        pairs.extend([(i, n) for n in neg_indices if threshold >= d[n]])
    return pairs


def mine_hard_triplets(model, data_loader, margin=0.5, batch_size=32):
    model.eval()
    # Compute pairwise distance
    new_loader = DataLoader(data_loader.dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            pin_memory=True if torch.cuda.is_available() else False)

    features, _ = extract_features(model, new_loader, print_freq=10)
    distmat = pairwise_distance(features)
    # len(features)
    distmat = distmat.cpu().numpy()
    # Get the pids
    dataset = data_loader.dataset.dataset
    pids = np.asarray([pid for _, pid, _ in dataset])
    # Find the hard triplets

    pids_exp = np.repeat(pids, pids.shape[0]).reshape(pids.shape[0], pids.shape[0])
    mask = (pids_exp == pids_exp.T)
    # return distmat, mask
    # distmat = distmat.reshape(-1)
    # mask = mask.reshape(-1)
    #
    # def get_topk_ind(arr, mask, k, ascend=True):
    #     gl_ind = np.arange(arr.shape[0], dtype=int)
    #     lc_ind = gl_ind[mask]
    #     lc_arr = arr[mask]
    #     if ascend:
    #         return lc_ind[np.argsort(lc_arr)[:k]]
    #     else:
    #         return lc_ind[np.argsort(lc_arr)[::-1][:k]]
    #
    # posind = get_topk_ind(distmat, mask, pids.shape[0], ascend=False)
    # negind = get_topk_ind(distmat, mask, pids.shape[0], ascend=True)
    #
    # triplets = []
    # for ind in posind:
    #     triplets.extend([
    #         ind // pids.shape[0], ind % (pids.shape[0])
    #     ])
    # for ind in negind:
    #     triplets.extend([
    #         ind // pids.shape[0], ind % pids.shape[0]
    #     ])

    triplets = []
    for i in np.random.permutation(range(len(distmat))):
        # print(i)
        d = distmat[i]
        pos_indices = np.where(pids == pids[i])[0]
        neg_indices = np.where(pids != pids[i])[0]
        sorted_pos = np.argsort(d[pos_indices])[::-1]
        for j in sorted_pos:
            p = pos_indices[j]
            mask = (d[neg_indices] <= d[p] + margin)
            neg_indices = neg_indices[mask]
            triplets.extend([(i, p, n) for n in neg_indices])
        if len(triplets) > pids.shape[0] * 3:
            break
    print('mined hard', len(triplets), 'num pids ', pids.shape[0])
    return triplets
