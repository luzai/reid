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


def compute_soft_hard_retrieval(distance_matrix, labels, label_batch=None):
    from chainer import cuda
    softs = []
    hards = []
    retrievals = []
    recalls = []
    if label_batch is None:
        label_batch = labels
    distance_matrix = cuda.to_cpu(distance_matrix)
    labels = cuda.to_cpu(labels)
    label_batch = cuda.to_cpu(label_batch)

    K = 1001  # "K" for top-K
    krange = [1, 2, 3, 4, 5, 10, 100, 1000]
    for d_i, label_i in zip(distance_matrix, label_batch):
        top_k_indexes = np.argpartition(d_i, K)[:K]
        sorted_top_k_indexes = top_k_indexes[np.argsort(d_i[top_k_indexes])]
        ranked_labels = labels[sorted_top_k_indexes]
        # 0th entry is excluded since it is always 0
        ranked_hits = ranked_labels[1:] == label_i
        n_true = (labels == label_i).sum() - 1
        # soft top-k, k = 1, 2, 5, 10
        soft = [np.any(ranked_hits[:k]) for k in krange]
        softs.append(soft)
        # hard top-k, k = 2, 3, 4
        hard = [np.all(ranked_hits[:k]) for k in krange]
        hards.append(hard)
        # retrieval top-k, k = 2, 3, 4
        retrieval = [np.mean(ranked_hits[:k]) for k in krange]
        retrievals.append(retrieval)
        recall = [np.sum(ranked_hits[:k]) / min(n_true, k) for k in krange]  # not recall not use
        recalls.append(recall)

    average_soft = np.array(softs).mean(axis=0)
    average_hard = np.array(hards).mean(axis=0)
    average_retrieval = np.array(retrievals).mean(axis=0)
    average_recall = np.array(recalls).mean(axis=0)
    return average_soft, average_hard, average_retrieval, average_recall


def Recall_at_ks(sim_mat, data='cub', query_ids=None, gallery_ids=None):
    # start_time = time.time()
    # print(start_time)
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids
    :param data

    Compute  [R@1, R@2, R@4, R@8]
    """

    ks_dict = dict()
    ks_dict['cub'] = [1, 2, 4, 8, 16, 32]
    ks_dict['car'] = [1, 2, 4, 8, 16, 32]
    ks_dict['jd'] = [1, 2, 4, 8]
    ks_dict['product'] = [1, 10, 100, 1000]
    ks_dict['shop'] = [1, 10, 20, 30, 40, 50]

    if data is None:
        data = 'cub'
    k_s = ks_dict[data]

    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape
    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)

    num_max = int(1e6)

    if m > num_max:
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:num_max]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = num_max

    # Hope to be much faster  yes!!
    num_valid = np.zeros(len(k_s))
    neg_nums = np.zeros(m)
    plt.figure()
    for i in range(m):
        x = sim_mat[i]
        score = x
        score -= score.min()
        score /= score.max()
        label = gallery_ids == query_ids[i]
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(label, score)
        # plt.step(recall, precision, alpha=0.2,where='post')
        # plt.show()
        # if i > 100: break
        pos_max = np.max(x[gallery_ids == query_ids[i]])
        neg_num = np.sum(x > pos_max)
        neg_nums[i] = neg_num
    # plt.show()
    for i, k in enumerate(k_s):
        if i == 0:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp
        else:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp - num_valid[i - 1]
    # t = time.time() - start_time
    # print(t)
    return num_valid / float(m)


def NMI(X, ground_truth, n_cluster=3):
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score
    # X = [to_numpy(x) for x in X]
    # list to numpy
    # X = np.array(X)
    X = to_numpy(X)
    ground_truth = np.asarray(ground_truth)
    # print('x_type:', type(X))
    # print('label_type:', type(ground_truth))
    kmeans = KMeans(n_clusters=n_cluster, n_jobs=-1, random_state=0).fit(X)

    print('K-means done')
    nmi = normalized_mutual_info_score(ground_truth, kmeans.labels_)
    return nmi


def eval_market1501_faiss(indices, q_pids, g_pids, q_camids, g_camids, max_rank=11):
    num_q, num_g = len(q_pids), len(g_pids)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

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
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # keep = np.invert(remove)
        keep = (g_pids[order] != q_pid) | (g_camids[order] != q_camid)
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
        if num_rel == 0:
            AP = 0
        else:
            AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    all_AP = np.asarray(all_AP)
    mAP = np.mean(all_AP)
    print('numq mAP shape', num_q, all_AP.shape)
    return mAP, all_cmc


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    if distmat.shape[0] < 65535:
        indices = indices.astype(np.uint16)
    else:
        indices = indices.astype(np.uint32)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

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
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # keep = np.invert(remove)
        keep = (g_pids[order] != q_pid) | (g_camids[order] != q_camid)
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
        if num_rel == 0:
            AP = 0
        else:
            AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    all_AP = np.asarray(all_AP)
    mAP = np.mean(all_AP)
    print('numq mAP shape', num_q, all_AP.shape)
    return mAP, all_cmc


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
            # lz.msgpack_dump([xx, yy], work_path + 'tmp.mp')
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

    def evaluate_recall(self, data_loader, query, gallery):
        self.model.eval()
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)

        features = extract_features(self.model, data_loader)[0]
        distmat, xx, yy = pairwise_distance(features, query, gallery, )

        # Recall_at_ks(-distmat - (-distmat).min(), 'cub', query_ids, gallery_ids)
        res = Recall_at_ks(-distmat, 'product', query_ids, gallery_ids)
        return res

    def evaluate_retrival(self, data_loader, query, gallery):
        logging.info('sart eval retrieval')
        self.model.eval()
        query_ids = [pid for _, pid, _ in query]
        # gallery_ids = [pid for _, pid, _ in gallery]
        query_ids = np.asarray(query_ids)
        # gallery_ids = np.asarray(gallery_ids)

        features = extract_features(self.model, data_loader)[0]
        x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        xx = x.view(x.size(0), -1).numpy()
        yy = y.view(y.size(0), -1).numpy()

        num_examples = len(xx)
        xp = np
        D_batches = []
        softs = []
        hards = []
        retrievals = []
        recalls = []
        y_data = yy
        c_data = query_ids
        yy = xp.sum(yy ** 2.0, axis=1)
        batch_size = 1024
        return_distance_matrix = False
        for start in range(0, num_examples, batch_size):
            end = start + batch_size
            if end > num_examples:
                end = num_examples
            y_batch = y_data[start:end]
            yy_batch = yy[start:end]
            c_batch = c_data[start:end]

            D_batch = yy + yy_batch[:, None] - 2.0 * xp.dot(y_batch, y_data.T)
            xp.maximum(D_batch, 0, out=D_batch)
            D_batch += 1e-40
            # ensure the diagonal components are zero
            xp.fill_diagonal(D_batch[:, start:end], 0)

            soft, hard, retr, recl = compute_soft_hard_retrieval(
                D_batch, c_data, c_batch)  # krange = [1, 2, 3, 4, 5, 10,100,1000 ]

            softs.append(len(y_batch) * soft)
            hards.append(len(y_batch) * hard)
            retrievals.append(len(y_batch) * retr)
            recalls.append(len(y_batch) * recl)
            if return_distance_matrix:
                D_batches.append(D_batch)

        avg_softs = xp.sum(softs, axis=0) / num_examples
        avg_hards = xp.sum(hards, axis=0) / num_examples
        avg_retrievals = xp.sum(retrievals, axis=0) / num_examples
        avg_recalls = xp.sum(recalls, axis=0) / num_examples
        logging.info(f'finish eval top1 {avg_softs[0]}')
        res = {}
        for ind, k in enumerate([1, 2, 3, 4, 5, 10, 100, 1000]):
            res[f'top-{k}-soft'] = avg_softs[ind]
            res[f'top-{k}-hard'] = avg_hards[ind]
            res[f'top-{k}-retr'] = avg_retrievals[ind]
            res[f'top-{k}-recl'] = avg_recalls[ind]
        res['top-1'] = res['top-1-soft']
        return res

    def evaluate(self, data_loader, query, gallery, metric=None, **kwargs):
        self.model.eval()
        query_ps = [path for path, _, _ in query]
        gallery_ps = [path for path, _, _ in gallery]
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
        # final = kwargs.get('final', False)
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
            lz.timer.since_last_check(f'start eval whether to use rerank :  {rerank}')
            ## todo dump feature only
            # msgpack_dump([features, query, gallery, ], work_path + 't.pk')
            # return 'ok'
            if self.conf == 'market1501':
                # try:
                #     ## use faiss
                #
                #     import faiss
                #
                #     xx = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
                #     yy = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
                #     xx = to_numpy(xx)
                #     yy = to_numpy(yy)
                #     index = faiss.IndexFlatL2(xx.shape[1])
                #     index.add(yy)
                #     _, ranklist = index.search(xx, yy.shape[0])
                #     # _, ranklist = index.search(xx, 11)
                #     mAP, all_cmc, all_AP = eval_market1501_wrap(
                #         np.asarray(ranklist), query_ids, gallery_ids, query_cams,
                #         gallery_cams,
                #         max_rank=10, faiss=True)
                #     lz.timer.since_last_check(f'faiss {mAP}')
                # except ImportError as e:

                ## use NN
                distmat, xx, yy = pairwise_distance(features, query, gallery, metric=metric, rerank=rerank)
                print(f'facing {distmat.shape}')
                distmat = distmat.astype(np.float16)
                mAP, all_cmc, all_AP = eval_market1501_wrap(
                    distmat, query_ids, gallery_ids, query_cams, gallery_cams,
                    max_rank=10)

                # nmi = NMI(xx, query_ids, 20)
                # print('mni is ', nmi)
                lz.timer.since_last_check(f'NN {mAP}')

                msgpack_dump([distmat, xx, yy, query, gallery, all_AP], work_path + 't.pk') # todo for plot
                if rerank:
                    res = lz.dict_concat([res,
                                          {'mAP.rk': mAP, 'top-1.rk': all_cmc[0], 'top-5.rk': all_cmc[4],
                                           'top-10.rk': all_cmc[9]}])
                else:
                    res = lz.dict_concat([res,
                                          {'mAP': mAP},
                                          {f'top-{ind+1}': v for ind, v in enumerate(all_cmc)},
                                          ])
            else:
                ## use NN  & mkt
                # distmat, xx, yy = pairwise_distance(features, query, gallery, metric=metric, rerank=rerank)
                # distmat = distmat.astype(np.float16)
                # mAP, all_cmc, all_AP = eval_market1501_wrap(
                #     distmat, query_ids, gallery_ids, query_cams, gallery_cams,
                #     max_rank=10)
                # print('--> Mean AP: {:4.1%}'.format(mAP), all_cmc )

                ## use faiss & mkt
                # import faiss
                # xx = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
                # yy = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
                # xx = to_numpy(xx)
                # yy = to_numpy(yy)
                # index = faiss.IndexFlatL2(xx.shape[1])
                # index.add(yy)
                # _, ranklist = index.search(xx, yy.shape[0])
                # # _, ranklist = index.search(xx, 11)
                # mAP, all_cmc, all_AP = eval_market1501_wrap(
                #     np.asarray(ranklist), query_ids, gallery_ids, query_cams,
                #     gallery_cams,
                #     max_rank=10, faiss=True)
                # lz.timer.since_last_check(f'faiss {mAP}')

                ## use NN & cu03 in fact cu03 should use this
                distmat, xx, yy = pairwise_distance(features, query, gallery, metric=metric, rerank=rerank)
                print(f'facing {distmat.shape}')
                mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)

                cmc_configs = {
                    'cuhk03': dict(separate_camera_set=True,
                                   single_gallery_shot=True,
                                   first_match_break=False),
                }
                cmc_configs = {k: v for k, v in cmc_configs.items() if k == self.conf}

                cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                        query_cams, gallery_cams, **params)
                              for name, params in cmc_configs.items()}
                print(f'--> mAP {mAP}  cmc-1 {self.conf} {cmc_scores[self.conf][0]} ')
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
        # json_dump(res, self.args.logs_dir + '/res.json')
        logging.info(f'eval finish res {res}')
        return res


def eval_market1501_wrap_ignore_cid(distmat, q_pids, g_pids, topk=10):
    q_camids = np.arange(len(q_pids))
    g_camids = np.arange(len(g_pids)) + q_camids.max()
    return eval_market1501_wrap(distmat, q_pids, g_pids, q_camids, g_camids, topk)

# if __name__ == '__main__':
#     def func():
#         distmat, q_pids, g_pids, q_camids, g_camids = lz.msgpack_load(work_path + 'tmp.mp')
#
#         mAP, all_cmc = eval_market1501_wrap(distmat,
#                                             q_pids,
#                                             g_pids,
#                                             q_camids,
#                                             g_camids, 10)
#         print(mAP, all_cmc)
#
#
#     import multiprocessing as mp
#
#     p = mp.Process(target=func)
#     p.start()
#     p.join()
