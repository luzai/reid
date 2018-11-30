import sys

sys.path.insert(0, '/data1/xinglu/prj/open-reid')

from lz import *
import lz
from reid.lib import eval_market1501_wrap

features, query, gallery = msgpack_load(work_path + 'large.pk')

xx = np.vstack([features[f] for f, _, _ in query], )
yy = np.vstack([features[f] for f, _, _ in gallery], )
# distractor = np.random.rand(500000,128)
# y = np.vstack((y, distractor))
# y.shape
distractors = yy[-500000:]
yy = yy[:-500000]

from reid.lib import eval_market1501_wrap

query_ids = [pid for _, pid, _ in query]
gallery_ids = [pid for _, pid, _ in gallery]
query_cams = [cam for _, _, cam in query]
gallery_cams = [cam for _, _, cam in gallery]

query_ids = np.asarray(query_ids)
gallery_ids = np.asarray(gallery_ids)
query_cams = np.asarray(query_cams)
gallery_cams = np.asarray(gallery_cams)


def myeval(xx, yy, quanty=False):
    import faiss
    if quanty:
        quantizer = faiss.IndexFlatL2(xx.shape[1])
        index = faiss.IndexIVFPQ(quantizer, xx.shape[1], yy.shape[0], 16, 8)
        index.train(yy)
        index.nprobe = xx.shape[0]
    else:
        index = faiss.IndexFlatL2(xx.shape[1])

    index.add(yy)
    _, ranklist = index.search(xx, yy.shape[0])
    #     _, ranklist = index.search(xx, 11)
    #     print(ranklist.max(), ranklist.shape)
    ranklist = np.array(ranklist, dtype=np.int64)
    mAP, all_cmc, all_AP = eval_market1501_wrap(
        np.asarray(ranklist), query_ids, gallery_ids, query_cams,
        gallery_cams,
        max_rank=10, faiss=True)
    del ranklist, index
    import gc
    gc.collect()
    return mAP, all_cmc


res = []
for ndis in np.asarray([0, 100, 200, 300, 400, 500]) * 1000:
    #     for n in range(10):
    yy_new = np.vstack((yy, distractors[np.random.permutation(500000)[:ndis]]))
    mAP, all_cmc, = myeval(xx, yy_new, quanty=False)
    res.append([mAP, all_cmc])
    #     break

import json

print(json.dumps(to_json_format(res, allow_np=False)))
msgpack_dump(res, 'res.pk')
