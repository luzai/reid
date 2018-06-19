# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport cython
cimport numpy as np
import numpy as np
from libcpp cimport bool as bool_t
import time
cpdef bbox_overlaps(
        cython.floating [:,:] boxes,
        cython.floating [:,:] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef:
        unsigned int N = boxes.shape[0]
        unsigned int K = query_boxes.shape[0]
        cython.floating[:,:] overlaps = np.zeros((N, K))
        cython.floating iw, ih, box_area
        cython.floating ua
        unsigned int k, n
    for k in range(K):
        box_area = (
                (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return np.asarray(overlaps)

cpdef eval_market1501_wrap(distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank):
    distmat = np.asarray(distmat,dtype=np.float32)
    q_pids = np.asarray(q_pids, dtype=np.int64)
    g_pids = np.asarray(g_pids , dtype=np.int64)
    q_camids=np.asarray(q_camids,dtype=np.int64)
    g_camids=np.asarray(g_camids, dtype=np.int64)
    return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

ctypedef unsigned short my_int
my_npint = np.uint16
# If the size of gallery exceed 65535, uncomment the following.
# ctypedef unsigned int my_int
# my_npint = np.uint32

cpdef eval_market1501(
        float[:,:] distmat,
        long[:] q_pids,
        long[:] g_pids,
        long[:] q_camids,
        long[:] g_camids,
        long max_rank,
):
    # return 0,0
    cdef:
        long num_q = distmat.shape[0], num_g = distmat.shape[1]

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    tic = time.time()
    cdef my_int[:,:] indices = np.argsort(distmat, axis=1).astype(my_npint)
    print('time indices', time.time()-tic)
    tic = time.time()
    cdef bool_t[:,:] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.uint8)
    print('time matches', time.time()-tic)
    tic = time.time()
    cdef float[:,:] all_cmc = np.empty((num_q,max_rank),dtype=np.float32)
    print('time all_cmc', time.time()-tic)
    cdef float[:] all_AP = np.zeros(num_q,dtype=np.float32)

    cdef:
        my_int q_pid, q_camid
        my_int[:] order=np.zeros(num_g,dtype=my_npint)
        bool_t[:] keep=np.zeros(num_g,dtype=np.uint8)

        my_int num_valid_q = 0, q_idx, idx
        # long[:] orig_cmc=np.zeros(num_g,dtype=np.int64)
        float[:] orig_cmc=np.zeros(num_g,dtype=np.float32)
        float[:] cmc=np.zeros(num_g,dtype=np.float32), tmp_cmc=np.zeros(num_g,dtype=np.float32)
        my_int num_orig_cmc=0
        float num_rel=0.
        float tmp_cmc_sum =0.
        # num_orig_cmc is the valid size of orig_cmc, cmc and tmp_cmc
        bool_t orig_cmc_flag=0

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        for idx in range(num_g):
            keep[idx] = (g_pids[order[idx]]!=q_pid) or (g_camids[order[idx]]!=q_camid)
        # compute cmc curve
        num_orig_cmc=0
        orig_cmc_flag=0
        for idx in range(num_g):
            if keep[idx]:
                orig_cmc[num_orig_cmc] = matches[q_idx][idx]
                num_orig_cmc +=1
                if matches[q_idx][idx]>1e-31:
                    orig_cmc_flag=1
        if not orig_cmc_flag:
            all_AP[q_idx]=-1
            # print('continue ', q_idx)
            # this condition is true when query identity does not appear in gallery
            continue
        my_cusum(orig_cmc,cmc,num_orig_cmc)
        for idx in range(num_orig_cmc):
            if cmc[idx] >1:
                cmc[idx] =1
        all_cmc[q_idx] = cmc[:max_rank]
        num_valid_q+=1

        # print('ori cmc', np.asarray(orig_cmc).tolist())
        # print('cmc', np.asarray(cmc).tolist())
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = 0.
        for idx in range(num_orig_cmc):
            num_rel += orig_cmc[idx]
        my_cusum( orig_cmc, tmp_cmc, num_orig_cmc)
        for idx in range(num_orig_cmc):
            tmp_cmc[idx] = tmp_cmc[idx] / (idx+1.) * orig_cmc[idx]
        # print('tmp_cmc', np.asarray(tmp_cmc).tolist())

        tmp_cmc_sum=my_sum(tmp_cmc,num_orig_cmc)
        all_AP[q_idx] = tmp_cmc_sum / num_rel
        # print('final',tmp_cmc_sum, num_rel, tmp_cmc_sum / num_rel,'\n')

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    # print_dbg('all ap', all_AP)
    # print_dbg('all cmc', all_cmc)
    all_AP_np = np.asarray(all_AP)
    all_AP_np[np.isclose(all_AP,-1)] = np.nan
    return  np.nanmean(all_AP_np), \
            np.asarray(all_cmc).astype(np.float32).sum(axis=0) / num_valid_q

def print_dbg(msg, val):
    print(msg, np.asarray(val))

cpdef void my_cusum(
        cython.numeric[:] src,
        cython.numeric[:] dst,
        long size
    ) nogil:
    cdef:
        long idx
    for idx in range(size):
        if idx==0:
            dst[idx] = src[idx]
        else:
            dst[idx] = src[idx]+dst[idx-1]

cpdef cython.numeric my_sum(
        cython.numeric[:] src,
        long size
) nogil:
    cdef:
        long idx
        cython.numeric ttl=0
    for idx in range(size):
        ttl+=src[idx]
    return ttl