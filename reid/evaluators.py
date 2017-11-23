from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader
from .utils.data.preprocessor import *
from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import *
from .utils.meters import AverageMeter


def extract_features(model, data_loader, print_freq=10, metric=None, output_file=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def extract_embeddings(model, data_loader, print_freq=10, ):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    embeddings = []
    # labels= OrderedDict()
    end = time.time()
    for i, inputs in enumerate(data_loader):
        data_time.update(time.time() - end)
        outputs = extract_cnn_embeddings(model, inputs)

        embeddings.append(outputs)
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Embedding: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'.format(
                i + 1, len(data_loader),
                batch_time.val, batch_time.avg,
                data_time.val, data_time.avg))

    return torch.cat(embeddings)


def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        print('feature size ', x.size())
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None and metric.algorithm != 'euclidean':
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10),
                 return_all=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,  # hard
                         single_gallery_shot=False,  # hard
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,  # hard
                           single_gallery_shot=False,  # hard
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    if return_all:
        return cmc_scores
    else:
        # return cmc_scores['allshots'][0]
        return cmc_scores['cuhk03'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
        self.distmat=None

    def evaluate(self, data_loader, query, gallery, metric=None, return_all=False, final=False):
        features, _ = extract_features(self.model, data_loader)
        if not final and len(query) > 2000:
            choice = np.random.choice(len(query), 2000)
            query = np.array(query)[choice]
            gallery = np.array(gallery)[choice]

        distmat = pairwise_distance(features, query, gallery, metric=metric)
        self.distmat=distmat.cpu().numpy()
        return evaluate_all(distmat, query=query, gallery=gallery, return_all=return_all)


class CascadeEvaluator(object):
    def __init__(self, base_model, embed_model, embed_dist_fn=None, ):
        super(CascadeEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.embed_dist_fn = embed_dist_fn
        self.distmat1 = self.distmat2=None

    def evaluate(self, data_loader, query, gallery, cache_file=None,
                 rerank_topk=100, return_all=False, cmc_topk=(1, 5, 10), one_stage=True):
        if one_stage:
            rerank_topk = len(gallery)
        # Extract features image by image
        features, _ = extract_features(self.base_model, data_loader,
                                       output_file=cache_file)

        # Compute pairwise distance and evaluate for the first stage
        distmat = pairwise_distance(features, query, gallery)
        print("First stage evaluation:")
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]

        cmc_configs = {
            'allshots': dict(separate_camera_set=False,  # hard
                             single_gallery_shot=False,  # hard
                             first_match_break=False),
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False),
            'market1501': dict(separate_camera_set=False,  # hard
                               single_gallery_shot=False,  # hard
                               first_match_break=True)}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        print('CMC Scores{:>12}{:>12}{:>12}'
              .format('allshots', 'cuhk03', 'market1501'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                  .format(k, cmc_scores['allshots'][k - 1],
                          cmc_scores['cuhk03'][k - 1],
                          cmc_scores['market1501'][k - 1]))

        # Sort according to the first stage distance
        distmat = distmat.cpu().numpy()
        self.distmat1=distmat
        rank_indices = np.argsort(distmat, axis=1)

        # Build a data loader for topk predictions for each query
        pair_samples = []
        for i, indices in enumerate(rank_indices):
            query_fname, _, _ = query[i]
            for j in indices[:rerank_topk]:
                gallery_fname, _, _ = gallery[j]
                pair_samples.append((query_fname, gallery_fname))

        data_loader = DataLoader(
            KeyValuePreprocessor(features),
            sampler=pair_samples,
            batch_size=min(len(gallery)*rerank_topk, 4096*2),
            num_workers=4, pin_memory=False)

        # Extract embeddings of each pair
        embeddings = extract_embeddings(self.embed_model, data_loader)
        if self.embed_dist_fn is not None:
            print('before embed_dist fn', embeddings.size())
            embeddings = self.embed_dist_fn(embeddings)
            print('after embed dist fn', embeddings.size())
        # type(embeddings), type(torch.autograd.Variable(embeddings))
        # Merge two-stage distances
        for k, embed in enumerate(embeddings):
            i, j = k // rerank_topk, k % rerank_topk
            distmat[i, rank_indices[i, j]] = embed.data.cpu().numpy()
        if not one_stage:
            for i, indices in enumerate(rank_indices):
                bar = max(distmat[i][indices[:rerank_topk]])
                gap = max(bar + 1. - distmat[i, indices[rerank_topk]], 0)
                if gap > 0:
                    distmat[i][indices[rerank_topk:]] += gap

        self.distmat2=distmat

        print("Second stage evaluation: (one stage?)", one_stage)

        cmc_scores2 = {name: cmc(distmat, query_ids, gallery_ids,
                                 query_cams, gallery_cams, **params)
                       for name, params in cmc_configs.items()}

        print('CMC Scores{:>12}{:>12}{:>12}'
              .format('allshots', 'cuhk03', 'market1501'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                  .format(k, cmc_scores2['allshots'][k - 1],
                          cmc_scores2['cuhk03'][k - 1],
                          cmc_scores2['market1501'][k - 1]))

        if return_all:
            return cmc_scores, cmc_scores2
        else:
            return cmc_scores['cuhk03'][0], cmc_scores2['cuhk03'][0]
