import lz
from lz import *
from collections import OrderedDict
from reid.utils.data.sampler import *
from reid.utils.data.preprocessor import *
from reid.evaluation_metrics import cmc, mean_ap
from reid.feature_extraction import *
from reid.utils.meters import AverageMeter
from reid.utils.rerank import *
from easydict import EasyDict as edict
from reid.lib.cython_eval import eval_market1501_wrap


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
        outputs = extract_cnn_feature(model,
                                      [imgs, npys])
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
                          data_time.val, data_time.avg), imgs.shape)

    print('Extract Features: [{}/{}]\t'
          'Time {:.3f} ({:.3f})\t'
          'Data {:.3f} ({:.3f})\t'
          .format(i + 1, len(data_loader),
                  batch_time.val, batch_time.avg,
                  data_time.val, data_time.avg), imgs.shape)
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


def parse_name(ds):
    args_ds = edict()
    if ds == 'cu03det':
        args_ds.dataset = 'cuhk03'
        args_ds.dataset_val = 'cuhk03'
        args_ds.dataset_mode = 'detect'
        args_ds.eval_conf = 'cuhk03'
    elif ds == 'cu03lbl':
        args_ds.dataset = 'cuhk03'
        args_ds.dataset_val = 'cuhk03'
        args_ds.dataset_mode = 'label'
        args_ds.eval_conf = 'cuhk03'
    elif ds == 'mkt' or ds == 'market' or ds == 'market1501':
        args_ds.dataset = 'market1501'
        args_ds.dataset_val = 'market1501'
        args_ds.eval_conf = 'market1501'
    elif ds == 'msmt':
        args_ds.dataset = 'msmt17'
        args_ds.dataset_val = 'market1501'
        args_ds.eval_conf = 'market1501'
    elif ds == 'cdm':
        args_ds.dataset = 'cdm'
        args_ds.dataset_val = 'market1501'
        args_ds.eval_conf = 'market1501'
    elif ds == 'viper':
        args_ds.dataset = 'viper'
        args_ds.dataset_val = 'viper'
        args_ds.eval_conf = 'market1501'
    elif ds == 'cu01hard':
        args_ds.dataset = 'cuhk01'
        args_ds.dataset_val = 'cuhk01'
        args_ds.eval_conf = 'cuhk03'
        args_ds.dataset_mode = 'hard'
    elif ds == 'cu01easy':
        args_ds.dataset = 'cuhk01'
        args_ds.dataset_val = 'cuhk01'
        args_ds.eval_conf = 'cuhk03'
        args_ds.dataset_mode = 'easy'
    elif ds == 'dukemtmc':
        args_ds.dataset = 'dukemtmc'
        args_ds.dataset_val = 'dukemtmc'
        args_ds.eval_conf = 'market1501'
    else:
        # raise ValueError(f'dataset ... {ds}')
        args_ds.dataset_val = ds
        args_ds.eval_conf = 'market1501'
    return args_ds


class Evaluator(object):
    def __init__(self, model, gpu=(0,), args=None, vid=False):
        super(Evaluator, self).__init__()
        self.model = model
        self.gpu = gpu
        self.distmat = None
        self.args = args
        self.vid = vid
        self.timer = lz.Timer()
        name_val = args.dataset_val or args.dataset
        self.args = args
        self.conf = parse_name(name_val).get('eval_conf', 'market1501')

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
                features = self.model(imgs)[0]
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
                features = self.model(imgs)[0]
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
        rerank = False  # todo
        print('!!rerank is ', rerank)
        if rerank:
            xx = to_numpy(qf)
            yy = to_numpy(gf)
            lz.msgpack_dump([xx, yy], work_path + 'tmp.mp')
            distmat = re_ranking(xx, yy)
        else:
            m, n = qf.size(0), gf.size(0)
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.numpy()

        if self.conf == 'market1501':
            self.timer.since_last_check('distmat ')
            # distmat = np.array(distmat, np.float16)
            print(f'facing {distmat.shape}')
            mAP, all_cmc = eval_market1501_wrap(distmat,
                                                q_pids,
                                                g_pids,
                                                q_camids,
                                                g_camids, 10)
            self.timer.since_last_check('map cmc ok')
            res = {'mAP': mAP, 'top-1': all_cmc[0], 'top-5': all_cmc[4], 'top-10': all_cmc[9]}

        else:
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

        features = extract_features(self.model, data_loader)[0]
        assert len(features) != 0
        res = {}
        final = kwargs.get('final', False)
        if 'prefix' in kwargs:
            prefix = kwargs.get('prefix', '') + '/'
        else:
            prefix = ''
        logging.info(f'prefix is {prefix}')
        if self.args['rerank']:
            rerank_range = [True, False]
        else:
            rerank_range = [False, ]
        for rerank in rerank_range:
            print('whether to use rerank :', rerank)
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
            logging.info('start eval')
            timer = lz.Timer()
            if self.conf == 'market1501':
                distmat = distmat.astype(np.float16)
                # del features
                print(f'facing {distmat.shape}')
                # if distmat.shape[0] < 10000:
                mAP, all_cmc = eval_market1501_wrap(
                    distmat, query_ids, gallery_ids, query_cams, gallery_cams,
                    max_rank=10)
                # else:
                #     mAP, all_cmc = eval_market1501(
                #         distmat, query_ids, gallery_ids, query_cams, gallery_cams,
                #         max_rank=10)
                if rerank:
                    res = lz.dict_concat([res,
                                          {'mAP.rk': mAP, 'top-1.rk': all_cmc[0], 'top-5.rk': all_cmc[4],
                                           'top-10.rk': all_cmc[9]}])
                else:
                    res = lz.dict_concat([res,
                                          {'mAP': mAP, 'top-1': all_cmc[0], 'top-5': all_cmc[4], 'top-10': all_cmc[9]}])
            else:
                mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
                print('Mean AP: {:4.1%}'.format(mAP))

                cmc_configs = {
                    'cuhk03': dict(separate_camera_set=True,
                                   single_gallery_shot=True,
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
            timer.since_start()
        # res['mAP'] += self.args.impr
        # res['top-1'] += self.args.impr
        # json_dump(res, self.args.logs_dir + '/res.json')
        return res


if __name__ == '__main__':
    def func():
        distmat, q_pids, g_pids, q_camids, g_camids = lz.msgpack_load(work_path + 'tmp.mp')

        mAP, all_cmc = eval_market1501_wrap(distmat,
                                            q_pids,
                                            g_pids,
                                            q_camids,
                                            g_camids, 10)
        print(mAP, all_cmc)


    import multiprocessing as mp

    p = mp.Process(target=func)
    p.start()
    p.join()
