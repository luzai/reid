# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport cython
cimport numpy as np
import numpy as np
from libcpp cimport bool as bool_t
import time

cpdef eval_market1501_wrap(distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank=10,
        faiss=False
           ):
    # distmat, q_pids, g_pids, q_camids, g_camids = list(map(lambda x:np.asarray(x) if x.flags.writeable else np.array(x),  [distmat, q_pids, g_pids, q_camids, g_camids]))
    distmat = np.array(distmat, dtype=np.float32)
    q_pids = np.array(q_pids, dtype=np.int64)
    g_pids = np.array(g_pids, dtype=np.int64)
    q_camids = np.array(q_camids, dtype=np.int64)
    g_camids = np.array(g_camids, dtype=np.int64)
    faiss = int(faiss)
    return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, faiss)


# ctypedef unsigned short my_int
# my_npint = np.uint16
# If the size of gallery exceed 65535, uncomment the following.
ctypedef unsigned int my_int
my_npint = np.uint32

cpdef eval_market1501(
        float[:,:] distmat,
        long[:] q_pids,
        long[:] g_pids,
        long[:] q_camids,
        long[:] g_camids,
        long max_rank,
        long faiss
):
    cdef:
        long num_q = distmat.shape[0], num_g = distmat.shape[1]
    # print('start eval')
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    cdef my_int[:,:] indices
    if faiss==0:
        indices = np.argsort(distmat, axis=1).astype(my_npint)
    else:
        indices = np.asarray(distmat, dtype = my_npint)
    cdef bool_t[:,:] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.uint8)
    tic = time.time()
    cdef float[:,:] all_cmc = np.zeros((num_q,max_rank), dtype=np.float32)
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
        float tmp_cmc_sum=0.
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

        # print_dbg('ori_cmc', orig_cmc)
        # print_dbg('cmc', cmc)
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = 0.
        for idx in range(num_orig_cmc):
            num_rel += orig_cmc[idx]
        my_cusum( orig_cmc, tmp_cmc, num_orig_cmc)
        for idx in range(num_orig_cmc):
            tmp_cmc[idx] = tmp_cmc[idx] / (idx+1.) * orig_cmc[idx]
        # print_dbg('tmp_cmc' , tmp_cmc)
        tmp_cmc_sum=my_sum(tmp_cmc,num_orig_cmc)
        all_AP[q_idx] = tmp_cmc_sum / num_rel
        # print('final',tmp_cmc_sum, num_rel, tmp_cmc_sum / num_rel,'\n')


    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    # print_dbg('all ap', all_AP)
    # print_dbg('all cmc', all_cmc)
    all_AP_np = np.asarray(all_AP)
    all_AP_np[np.isclose(all_AP,-1)] = np.nan
    return  np.nanmean(all_AP_np), \
            np.asarray(all_cmc).astype(np.float32).sum(axis=0) / num_valid_q, \
            all_AP_np


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
