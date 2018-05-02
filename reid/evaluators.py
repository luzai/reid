from collections import OrderedDict
from .utils.data.sampler import *
from torch.utils.data import DataLoader
from .utils.data.preprocessor import *
from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import *
from .utils.meters import AverageMeter
from reid.utils.rerank import *
import cvbase as cvb


def extract_features(model, data_loader, print_freq=1, limit=None, ):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    print('extract feature')
    end = time.time()
    for i, data in enumerate(data_loader):
        imgs, npys, fnames, pids = data.get('img'), data.get('npy'), data.get('fname'), data.get('pid')
        data_time.update(time.time() - end)
        # import gc
        # gc.collect()
        outputs = extract_cnn_feature(model,
                                      [imgs, npys])
        for fname, output, pid in zip(fnames, outputs, pids):
            # if limit is not None and int(pid) not in limit.tolist():
            #     continue
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

    print('Extract Features: [{}/{}]\t'
          'Time {:.3f} ({:.3f})\t'
          'Data {:.3f} ({:.3f})\t'
          .format(i + 1, len(data_loader),
                  batch_time.val, batch_time.avg,
                  data_time.val, data_time.avg))
    print(f'{len(features)} features, each of len {features.values().__iter__().__next__().shape[0]}')
    return features, labels


def extract_embeddings(model, data_loader, print_freq=10, ):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    embeddings = []
    print('extract embedding')
    end = time.time()
    for i, inputs in enumerate(data_loader):
        data_time.update(time.time() - end)
        outputs = extract_cnn_embeddings(model, inputs)
        # print(outputs.shape)
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

    print('Extract embedding: [{}/{}]\t'
          'Time {:.3f} ({:.3f})\t'
          'Data {:.3f} ({:.3f})\t'
          .format(i + 1, len(data_loader),
                  batch_time.val, batch_time.avg,
                  data_time.val, data_time.avg))
    res = torch.cat(embeddings)
    print(res.shape)
    return res


def pairwise_distance(features, query=None, gallery=None, metric=None, rerank=False):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        print('feature size ', x.size())
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, x, x.t())
        dist = dist.clamp(min=1e-12).sqrt()
        raise ValueError('todo')
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    if rerank:
        xx = to_numpy(x)
        yy = to_numpy(y)
        if xx.shape[0] > 2000:
            mem_save = True
        else:
            mem_save = False
        dist = re_ranking(xx, yy, mem_save)
        return dist, xx, yy
    else:
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        if metric is not None and metric.algorithm != 'euclidean':
            x = metric.transform(x)
            y = metric.transform(y)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return to_numpy(dist), to_numpy(x), to_numpy(y)


def query_to_df(query):
    return pd.DataFrame(query, columns=['fns', 'pids', 'cids'])


class Evaluator(object):
    def __init__(self, model, gpu=(0,), conf='cuhk03', args=None):
        super(Evaluator, self).__init__()
        self.model = model
        self.gpu = gpu
        self.distmat = None
        self.conf = conf
        self.args = args

    def evaluate(self, data_loader, query, gallery, metric=None, epoch=None, **kwargs):
        timer = cvb.Timer()
        timer.start()
        self.model.eval()
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]

        features, _ = extract_features(self.model, data_loader)
        assert len(features) != 0
        res = {}

        # for rerank in [False, True]:
        for rerank in [False, ]:
            # try:
            distmat, xx, yy = pairwise_distance(features, query, gallery, metric=metric, rerank=rerank)
            # except Exception as e:
            #     logging.error(e)
            #     continue
            db_name = self.args.logs_dir.split('/')[-1] + '.h5'
            with lz.Database(db_name) as db:
                for name in ['distmat', 'query_ids', 'gallery_ids', 'query_cams', 'gallery_cams']:
                    if rerank:
                        db['rk/' + name] = eval(name)
                    else:
                        db[name] = eval(name)
                db['smpl'] = xx
            # with pd.HDFStore(db_name) as db:
            #     db['query'] = query_to_df(query)
            #     db['gallery'] = query_to_df(gallery)

            with lz.Database(db_name) as db:
                print(list(db.keys()))
            # try:
            mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
            # except Exception as e:
            #     logging.error(e)
            #     continue
            print('Mean AP: {:4.1%}'.format(mAP))

            cmc_configs = {
                'cuhk03': dict(separate_camera_set=True,
                               single_gallery_shot=True,
                               first_match_break=False),
                'market1501': dict(separate_camera_set=False,  # hard
                                   single_gallery_shot=False,  # hard
                                   first_match_break=True),
                'allshots': dict(separate_camera_set=False,  # hard
                                 single_gallery_shot=False,  # hard
                                 first_match_break=False),
            }
            cmc_configs = {k: v for k, v in cmc_configs.items() if k == self.conf}
            cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                    query_cams, gallery_cams, **params)
                          for name, params in cmc_configs.items()}
            print(f'cmc-1 {self.conf} {cmc_scores[self.conf][0]} ')
            if rerank:
                res = lz.dict_concat([res,
                                      {'mAP.rk': mAP,
                                       'top-1.rk': cmc_scores[self.conf][0],
                                       'top-5.rk': cmc_scores[self.conf][4],
                                       'top-10.rk': cmc_scores[self.conf][9],
                                       }])
            else:
                res = lz.dict_concat([res,
                                      {'mAP': mAP,
                                       'top-1': cmc_scores[self.conf][0],
                                       'top-5': cmc_scores[self.conf][4],
                                       'top-10': cmc_scores[self.conf][9],
                                       }])
        return res


class CascadeEvaluator(object):
    def __init__(self, base_model, embed_model, embed_dist_fn=None, ):
        super(CascadeEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.embed_dist_fn = embed_dist_fn
        self.distmat1 = self.distmat2 = None

    def evaluate(self, data_loader, query, gallery,
                 rerank_topk=100, return_all=False, cmc_topk=(1, 5, 10),
                 one_stage=True, need_second=True):
        self.base_model.eval()
        self.embed_model.eval()
        if one_stage:
            rerank_topk = len(gallery)
        # Extract features image by image
        features, _ = extract_features(self.base_model, data_loader, )

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

        print('CMC Scores|{:>12}|{:>12}|{:>12}'
              .format('allshots', 'cuhk03', 'market1501'))
        print('--|--|--|--')
        for k in cmc_topk:
            print('  top-{:<4}|{:12.1%}|{:12.1%}|{:12.1%}'
                  .format(k, cmc_scores['allshots'][k - 1],
                          cmc_scores['cuhk03'][k - 1],
                          cmc_scores['market1501'][k - 1]))

        if not need_second:
            if return_all:
                return cmc_scores, 0
            else:
                return cmc_scores['cuhk03'][0], 0

        # Sort according to the first stage distance
        distmat = to_numpy(distmat)
        self.distmat1 = distmat
        rank_indices = np.argsort(distmat, axis=1)

        # Build a data loader for topk predictions for each query
        pair_samples = []
        for i, indices in enumerate(rank_indices):
            query_fname, _, _ = query[i]
            for j in indices[:rerank_topk]:
                gallery_fname, _, _ = gallery[j]
                pair_samples.append((query_fname, gallery_fname))
        len(features)
        data_loader = DataLoader(
            KeyValuePreprocessor(features),
            sampler=pair_samples,
            batch_size=min(len(gallery) * rerank_topk, 1024 * 4),
            num_workers=4, pin_memory=False)
        # features.values().__iter__().__next__()
        # Extract embeddings of each pair
        embeddings = extract_embeddings(self.embed_model, data_loader)
        if self.embed_dist_fn is not None:
            print('before embed_dist fn', embeddings.size())
            embeddings = self.embed_dist_fn(embeddings)
            print('after embed dist fn', embeddings.size())

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

        self.distmat2 = distmat

        print("Second stage evaluation: (one stage?)", one_stage)

        cmc_scores2 = {name: cmc(distmat, query_ids, gallery_ids,
                                 query_cams, gallery_cams, **params)
                       for name, params in cmc_configs.items()}

        print('CMC Scores|{:>12}|{:>12}|{:>12}'
              .format('allshots', 'cuhk03', 'market1501'))
        print('--|--|--|--')
        for k in cmc_topk:
            print('  top-{:<4}|{:12.1%}|{:12.1%}|{:12.1%}'
                  .format(k, cmc_scores2['allshots'][k - 1],
                          cmc_scores2['cuhk03'][k - 1],
                          cmc_scores2['market1501'][k - 1]))

        # from lz import *
        # plt.matshow(distmat)
        # plt.colorbar()
        #
        # plt.matshow(np.log(distmat) )
        # plt.colorbar()
        # plt.show()

        if return_all:
            return cmc_scores, cmc_scores2
        else:
            return cmc_scores['cuhk03'][0], cmc_scores2['cuhk03'][0]


class SiameseEvaluator(object):
    def __init__(self, base_model, embed_model, embed_dist_fn=None):
        super(SiameseEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.embed_dist_fn = embed_dist_fn

    def evaluate(self, data_loader, query, gallery, cache_file=None):
        # Extract features image by image
        features, _ = extract_features(self.base_model, data_loader, )
        if cache_file is not None:
            features = FeatureDatabase(cache_file, 'r')

        # Build a data loader for exhaustive (query, gallery) pairs
        query_keys = [fname for fname, _, _ in query]
        gallery_keys = [fname for fname, _, _ in gallery]
        data_loader = DataLoader(
            KeyValuePreprocessor(features),
            sampler=ExhaustiveSampler((query_keys, gallery_keys,),
                                      return_index=False),
            batch_size=min(len(gallery), 4096 * 8),
            num_workers=4, pin_memory=False)

        # Extract embeddings of each (query, gallery) pair
        embeddings = extract_embeddings(self.embed_model, data_loader)
        if self.embed_dist_fn is not None:
            embeddings = self.embed_dist_fn(embeddings)

        if cache_file is not None:
            features.close()

        # Convert embeddings to distance matrix
        distmat = embeddings.contiguous().view(len(query), len(gallery))
        # cvb.dump(distmat.cpu(),'dbg.pkl')
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]

        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
        print('Mean AP: {:4.1%}'.format(mAP))

        # Evaluate CMC scores
        mcmc1 = cmc(distmat, query_ids, gallery_ids,
                    query_cams, gallery_cams,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)[0]
        ccmc1 = cmc(distmat, query_ids, gallery_ids,
                    query_cams, gallery_cams,
                    separate_camera_set=True,
                    single_gallery_shot=True,
                    first_match_break=False)[0]
        print('market1501 cmc-1', mcmc1, 'cu cmc-1', ccmc1)

        return mAP, mcmc1
