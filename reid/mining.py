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

def _stat(ndarr):
    return ndarr.shape, np.unique(ndarr).shape

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
    n = 1024*4  # batch_size
    distmat_n = distmat.copy()
    distmat_n[mask == True] = distmat_n.max()
    no_sel = np.setdiff1d(np.arange(pids.shape[0]), np.random.choice(pids.shape[0], 1024*4))
    distmat_n[no_sel, :] = distmat_n.max()
    distmat_n[:, no_sel] = distmat_n.max()
    ind = np.argpartition(distmat_n.ravel(), n)[:n]
    ind = ind[np.argsort(distmat_n.ravel()[ind])]
    ind = ind[3:256 + 3]
    # ind = np.random.choice(ind, n)
    # plt.plot(np.sort(distmat_n.ravel()[ind]),'.')

    # plt.hist(distmat_n.ravel()[ind] )
    anc, neg = np.unravel_index(ind, distmat.shape)
    _stat(anc), _stat(neg), _stat(np.concatenate((anc,neg)))
    triplets = []
    for anc_, neg_ in zip(anc, neg):
        # pos_inds = np.where(pids == pids[anc_])[0]
        pos_inds = np.where(mask[anc_] == True)[0]
        pos_ind = np.random.choice(pos_inds)
        d = distmat[anc_][pos_inds]

        triplets.append([anc_, pos_ind, neg_])

    # cvb.dump(distmat, 'distmat.pkl')
    # cvb.dump(mask, 'mask.pkl')
    # db=Database('fea.h5')
    # features_id2key= dict(zip(range(len(features.keys())), features.keys()))
    # for ind_ in np.asarray(triplets).ravel():
    #     key=features_id2key[ind_]
    #     db[key] = to_numpy(features[key])
    # db.close()
    print('mined hard', len(triplets), 'num pids ', pids.shape[0])
    return triplets
