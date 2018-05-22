from collections import OrderedDict
from reid.utils.data.sampler import *
from reid.utils.data.preprocessor import *
from reid.evaluation_metrics import cmc, mean_ap
from reid.feature_extraction import *
from reid.utils.meters import AverageMeter
from reid.utils.rerank import *


def extract_features(model, data_loader, print_freq=1):
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


def pairwise_distance(features, query=None, gallery=None, metric=None, rerank=False, vid=False):
    if query is None and gallery is None:
        # n = len(features)
        # x = torch.cat(list(features.values()))
        # x = x.view(n, -1)
        # print('feature size ', x.size())
        # if metric is not None:
        #     x = metric.transform(x)
        # dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, x, x.t())
        # dist = dist.clamp(min=1e-12).sqrt()
        raise ValueError('todo')
        # return dist
    if vid:
        x = torch.cat([features[tuple(f)].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([features[tuple(f)].unsqueeze(0) for f, _, _ in gallery], 0)
    else:
        x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    if rerank:
        xx = to_numpy(x)
        yy = to_numpy(y)
        # if xx.shape[0] > 2000:
        #     mem_save = True
        # else:
        #     mem_save = False
        # logging.info(f'mem saving mode {mem_save}')

        dist = re_ranking(xx, yy)
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
    def __init__(self, model, gpu=(0,), conf='cuhk03', args=None, vid=False):
        super(Evaluator, self).__init__()
        self.model = model
        self.gpu = gpu
        self.distmat = None
        self.conf = conf
        self.args = args
        self.vid = vid
        self.timer = lz.Timer()

    def evaluate_vid(self, queryloader, galleryloader, metric=None, **kwargs):
        self.model.eval()
        res = {}
        self.timer.since_last_check('start eval')
        with torch.no_grad():
            qf, q_pids, q_camids = [], [], []
            for batch_idx, data in enumerate(queryloader):
                (imgs, pids, camids) = data['img'], data['pid'], data['cid']
                imgs = imgs.cuda()
                b, s, c, h, w = imgs.size()
                imgs = imgs.view(b * s, c, h, w)
                features, _ = self.model(imgs)
                features = features.view(b, s, -1)
                features = torch.mean(features, 1)  # use avg
                features = features.data.cpu()
                qf.append(features)
                q_pids.extend(pids)
                q_camids.extend(camids)
            qf = torch.cat(qf, 0)
            q_pids = np.asarray(q_pids)
            q_camids = np.asarray(q_camids)

            print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
            self.timer.since_last_check('extract query')

            gf, g_pids, g_camids = [], [], []
            for batch_idx, data in enumerate(galleryloader):
                (imgs, pids, camids) = data['img'], data['pid'], data['cid']
                imgs = imgs.cuda()
                b, s, c, h, w = imgs.size()
                imgs = imgs.view(b * s, c, h, w)
                features, _ = self.model(imgs)
                features = features.view(b, s, -1)
                features = torch.mean(features, 1)
                features = features.data.cpu()
                gf.append(features)
                g_pids.extend(pids)
                g_camids.extend(camids)
            gf = torch.cat(gf, 0)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)

            print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
            self.timer.since_last_check('extract gallery')

        print("Computing distance matrix")
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()
        rerank = False
        self.timer.since_last_check('distmat ')
        print("Computing CMC and mAP")
        mAP = mean_ap(distmat, q_pids, g_pids, q_camids, g_camids)
        self.timer.since_last_check('mAP ok ')
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
        cmc_scores = {name: cmc(distmat, q_pids, g_pids,
                                q_camids, g_camids, **params)
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

        json_dump(res, self.args.logs_dir + '/res.json', 'w')
        self.timer.since_last_check('cmc ok')

        return res

    def evaluate(self, data_loader, query, gallery, metric=None, **kwargs):

        self.model.eval()
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        features, _ = extract_features(self.model, data_loader)
        assert len(features) != 0
        res = {}
        final = kwargs.get('final', False)
        if 'prefix' in kwargs:
            prefix = kwargs.get('prefix', '') + '/'
        else:
            prefix = ''
        logging.info(f'prefix is {prefix}')
        # if final:
        #     rerank_range = [False, True]
        # else:
        rerank_range = [False, ]
        for rerank in rerank_range:
            distmat, xx, yy = pairwise_distance(features, query, gallery, metric=metric, rerank=rerank)

            if final:
                db_name = self.args.logs_dir + '/' + self.args.logs_dir.split('/')[-1] + '.h5'
                with lz.Database(db_name) as db:
                    for name in ['distmat', 'query_ids', 'gallery_ids', 'query_cams', 'gallery_cams']:
                        if rerank:
                            db[prefix + 'rk/' + name] = eval(name)
                        else:
                            db[prefix + name] = eval(name)
                    db[prefix + 'smpl'] = xx
                # with pd.HDFStore(db_name) as db:
                #     db['query'] = query_to_df(query)
                #     db['gallery'] = query_to_df(gallery)

                with lz.Database(db_name) as db:
                    print(list(db.keys()))

            # mAP, all_cmc = eval_market1501(distmat, query_ids, gallery_ids, query_cams, gallery_cams, max_rank=10)
            # res = {'mAP': mAP, 'top-1': all_cmc[0], 'top-5': all_cmc[4], 'top-10': all_cmc[9]}

            mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
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

        # json_dump(res, self.args.logs_dir + '/res.json')
        return res


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return mAP, all_cmc


if __name__ == '__main__':
    with lz.Database(lz.root_path + '/exps/work/eval/eval.h5', 'r') as db:
        print(list(db.keys()))
        for name in ['distmat', 'query_ids', 'gallery_ids', 'query_cams', 'gallery_cams']:
            locals()[name] = db['test/' + name]
        print(distmat.shape)
        timer = lz.Timer()
        eval_market1501(distmat, query_ids, gallery_ids, query_cams, gallery_cams, 10)
        timer.since_start()
