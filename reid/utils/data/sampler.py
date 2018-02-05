from __future__ import absolute_import
from collections import defaultdict
import itertools
import numpy as np
import torch
import lz
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)
import queue
import pandas as pd


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4, batch_size=160, shuffle=True):
        assert batch_size % num_instances == 0
        self.batch_size = batch_size

        self.data_source = data_source
        self.num_instances = num_instances

        pids = np.asarray(data_source)[:, 1].astype(int)
        inds = np.arange(pids.shape[0], dtype=int)

        self.info = pd.DataFrame.from_items([
            ('pids', pids),
            ('inds', inds)
        ])
        self.num_samples = np.unique(pids).shape[0]

        self.shuffle = shuffle
        self.queue = queue.Queue()

    def __len__(self):
        return self.batch_size * 43

    def __iter__(self):
        # indices = torch.randperm(self.num_samples)
        ind_ind = 0
        grouped = self.info.groupby('pids')
        while ind_ind < len(self) // self.num_instances:
            if not self.queue.empty():
                yield self.queue.get()
            else:
                pid = np.random.choice(np.unique(self.info['pids']))
                dft = grouped.get_group(pid)
                t = dft['inds'].tolist()
                if len(t) >= self.num_instances:
                    t = np.random.choice(t, size=self.num_instances, replace=False)
                else:
                    t = np.random.choice(t, size=self.num_instances, replace=True)
                [self.queue.put(t_) for t_ in t]
                ind_ind += 1
                yield self.queue.get()


class RandomIdentityWeightedSampler(Sampler):
    def __init__(self, data_source, num_instances=4, batch_size=128,
                 rand_ratio=1.,
                 weights=None, subsample=None):
        assert batch_size % num_instances == 0
        self.batch_size = batch_size
        self.subsample = np.unique(np.asarray(subsample).ravel())
        self.data_source = data_source
        self.num_instances = num_instances
        self.rand_ratio = rand_ratio

        pids = np.asarray(data_source)[:, 1].astype(int)
        # data_source is img_path pida, cids
        inds = np.arange(pids.shape[0], dtype=int)
        if weights is None:
            weights = np.ones_like(inds, dtype=float)
            weights = weights / weights.sum()

        self.info = pd.DataFrame.from_items([
            ('pids', pids),
            ('inds', inds),
            ('probs', weights)
        ])
        self.pids = np.unique(pids)
        self.num_pids = self.pids.shape[0]
        self.inds = inds
        self.num_inds = self.info.shape[0]
        self.queue = queue.Queue()

    def __len__(self):
        return len(self.data_source)
        # return self.batch_size*43

    def get_batch_pids(self):
        pids = []
        with lz.Database('dop.h5') as db:
            dop = db['dop'].copy()
        lz.logging.debug('get new inds, {} {}'.format(dop, np.count_nonzero(dop == -1)))
        pids_now = np.random.choice(self.pids,
                                    size=int(self.batch_size / self.num_instances * self.rand_ratio),
                                    replace=False)
        pids.extend(pids_now.tolist())
        while len(pids) < self.batch_size / self.num_instances:
            pids_next = []
            for pid in pids_now:
                if dop[pid] == -1:
                    pids_next.extend(np.random.choice(self.pids, size=(1,)).tolist())
                else:
                    pids_next.append(dop[pid])
            pids.extend(pids_next)
            pids_now = pids_next

        return pids

    def get_batch_inds(self):
        inds = []
        groupes = self.info.groupby('pids')
        pids = self.get_batch_pids()
        for pid in pids:
            inds.extend(
                np.random.choice(
                    groupes.get_group(pid)['inds'].tolist(),
                    size=(self.num_instances,)
                ).tolist()
            )
        inds = inds[:self.batch_size]
        return inds

    def __iter__(self):
        cnt = 0
        while cnt < len(self):
            inds = self.get_batch_inds()
            for ind in inds:
                cnt += 1
                yield ind

        # ind_ind = 0
        # grouped = self.info.groupby('pids')
        # while ind_ind < len(self) // self.num_instances:
        #     if not self.queue.empty():
        #         tobe = self.queue.get()
        #         self.cache_ind.append(tobe)
        #         yield tobe
        #     else:
        #         probs = grouped.sum()['probs']
        #         pid = np.random.choice(probs.index, )
        #         pid = int(pid)
        #
        #         dft = grouped.get_group(pid)
        #         t = dft['inds'].tolist()
        #         probs_probs = np.asarray(dft['probs'], dtype=float)
        #         probs_probs = probs_probs / probs_probs.sum()
        #         if len(t) >= self.num_instances:
        #             t = np.random.choice(t, size=self.num_instances, replace=False, p=probs_probs)
        #         else:
        #             t = np.random.choice(t, size=self.num_instances, replace=True, p=probs_probs)
        #         # with self.queue.mutex:
        #         [self.queue.put(t_) for t_ in t]
        #         ind_ind += 1

    def update_weight(self, weights):
        self.info['probs'] = weights


def _choose_from(start, end, excluding=None, size=1, replace=False):
    num = end - start + 1
    if excluding is None:
        return np.random.choice(num, size=size, replace=replace) + start
    ex_start, ex_end = excluding
    num_ex = ex_end - ex_start + 1
    num -= num_ex
    inds = np.random.choice(num, size=size, replace=replace) + start
    inds += (inds >= ex_start) * num_ex
    return inds


class ExhaustiveSampler(Sampler):
    def __init__(self, data_sources, return_index=False):
        self.data_sources = data_sources
        self.return_index = return_index

    def __iter__(self):
        if self.return_index:
            return itertools.product(
                *[range(len(x)) for x in self.data_sources])
        else:
            return itertools.product(*self.data_sources)

    def __len__(self):
        return np.prod([len(x) for x in self.data_sources])


class RandomPairSampler(Sampler):
    def __init__(self, data_source, neg_pos_ratio=1):
        super(RandomPairSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.neg_pos_ratio = neg_pos_ratio
        # Sort by pid
        indices = np.argsort(np.asarray(data_source)[:, 1])
        self.index_map = dict(zip(np.arange(self.num_samples), indices))
        # Get the range of indices for each pid
        self.index_range = defaultdict(lambda: [self.num_samples, -1])
        for i, j in enumerate(indices):
            _, pid, _ = data_source[j]
            self.index_range[pid][0] = min(self.index_range[pid][0], i)
            self.index_range[pid][1] = max(self.index_range[pid][1], i)

    def __iter__(self):
        indices = np.random.permutation(self.num_samples)
        for i in indices:
            # anchor sample
            anchor_index = self.index_map[i]
            _, pid, _ = self.data_source[anchor_index]
            # positive sample
            start, end = self.index_range[pid]
            pos_index = _choose_from(start, end, excluding=(i, i))[0]
            yield anchor_index, self.index_map[pos_index]
            # negative samples
            neg_indices = _choose_from(0, self.num_samples - 1,
                                       excluding=(start, end),
                                       size=self.neg_pos_ratio)
            for neg_index in neg_indices:
                yield anchor_index, self.index_map[neg_index]

    def __len__(self):
        return self.num_samples * (1 + self.neg_pos_ratio)


class RandomTripletSampler(Sampler):
    def __init__(self, data_source):
        super(RandomTripletSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        # Sort by pid
        indices = np.argsort(np.asarray(data_source)[:, 1])
        self.index_map = dict(zip(np.arange(self.num_samples), indices))
        # Get the range of indices for each pid
        self.index_range = defaultdict(lambda: [self.num_samples, -1])
        for i, j in enumerate(indices):
            _, pid, _ = data_source[j]
            self.index_range[pid][0] = min(self.index_range[pid][0], i)
            self.index_range[pid][1] = max(self.index_range[pid][1], i)

    def __iter__(self):
        indices = np.random.permutation(self.num_samples)
        for i in indices:
            # anchor sample
            anchor_index = self.index_map[i]
            _, pid, _ = self.data_source[anchor_index]
            # positive sample
            start, end = self.index_range[pid]
            pos_index = _choose_from(start, end, excluding=(i, i))[0]
            pos_index = self.index_map[pos_index]
            # negative samples
            neg_index = _choose_from(0, self.num_samples - 1,
                                     excluding=(start, end))[0]
            neg_index = self.index_map[neg_index]
            yield anchor_index, pos_index, neg_index

    def __len__(self):
        return self.num_samples
