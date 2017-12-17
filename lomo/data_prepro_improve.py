import os
import pdb

print('current work dir: %s' % os.getcwd())
import sys

import cv2
import numpy as np

import random
import time

import argparse
import pdb

from multiprocessing import Pool
from threading import Thread


# from mlab.releases import R2014b as mlab
# mlab.path(mlab.path(),'/home/maochaojie/work/LOMO_XQDA/LOMO_XQDA/code')
# mlab.path(mlab.path(),'/home/maochaojie/work/LOMO_XQDA/LOMO_XQDA/bin')

# video_dict ={'key':'frames'=[],'fames_p'=[],'reshpe'=(h,w),'crop'=(h,w),'label'=[]}
# def lomo(data_in):
#     data_in_lomo = cv2.resize(data_in, (48,128))
#     lomo_des = mlab.IMGLOMO(data_in_lomo)
#     # print(len(lomo_des))
#     return lomo_des

def transform(I, rows, cols, pts1, pts2):
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(I, M, (cols, rows))
    return dst


def affine(M, N):
    ratio = 0.05
    ratio_s = 0.05

    dx = random.uniform(-ratio, ratio) * N
    dy = random.uniform(-ratio, ratio) * M
    ds = random.uniform(-ratio_s, ratio_s)
    ds_x = (N - (1 + ds) * N) / 2
    ds_y = (M - (1 + ds) * M) / 2
    if random.uniform(0, 1) > 0.3:
        pts2 = np.float32([[dx + ds_x, dy + ds_y], [N + dx - ds_x, dy + ds_y], [dx + ds_x, M + dy - ds_y]])
    else:
        pts2 = np.float32([[N + dx - ds_x, dy + ds_y], [dx + ds_x, dy + ds_y], [N + dx - ds_x, M + dy - ds_y]])

    return pts2


def processImage(filepath, newImgroot, Imgroot, newNpyroot, rf=False):
    #    print(Imgroot+filepath)
    data_in = cv2.imread(Imgroot + filepath)
    filename = filepath.split('/')[-1]
    new_Imgroot = newImgroot
    # print(data_in)
    # pdb.set_trace()
    if rf:
        folder = filepath.split('/')[-2]
        new_Imgroot = newImgroot + '%s/' % folder
        # newNpyroot = newNpyroot + '%s/'%folder
        if not os.path.exists(new_Imgroot):
            os.makedirs(new_Imgroot)
        # if not os.path.exists(newNpyroot):
        # os.mkdir(newNpyroot)

    newImgpath = new_Imgroot + '%s_%d.%s' % (filename.split('.')[0], 0, filename.split('.')[-1])

    print(newImgpath)
    cv2.imwrite(newImgpath, data_in)
    # newNpypath = newNpyroot+'%s_%d.%s'%(filename.split('.')[0], 0, 'npy')
    # lomo_des = lomo(data_in)
    # np.save(newNpypath,lomo_des)

    M, N, c = data_in.shape
    pts1 = np.float32([[0, 0], [N, 0], [0, M]])
    for i in range(5):
        pts2 = affine(M, N)
        data_out = transform(data_in, M, N, pts1, pts2)
        newpath = new_Imgroot + '%s_%d.%s' % (filename.split('.')[0], i + 1, filename.split('.')[-1])
        cv2.imwrite(newpath, data_out)


#	print(data_out)
# newNpypath = newNpyroot+'%s_%d.%s'%(filename.split('.')[0], i+1, 'npy')
# lomo_des = lomo(data_out)
# len(lomo_des)
# np.save(newNpypath,lomo_des)

class processor(object):
    def __init__(self, newImgroot, newNpyroot, Imgroot, keepfolder):
        self.newImgroot = newImgroot
        self.newNpyroot = newNpyroot
        self.Imgroot = Imgroot
        self.keepfolder = keepfolder

    def __call__(self, filepath):
        return processImage(filepath, self.newImgroot, self.Imgroot, self.newNpyroot, self.keepfolder)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--renamefolder', help='keep folder', type=bool, default=True)
    parser.add_argument('-I', '--newImgroot', help='root of new Img. End with /', default='./')
    parser.add_argument('-N', '--newNpyroot', help='root of new npy. End with /', default='./')
    parser.add_argument('-f', '--filelistpath', help='list of img')
    parser.add_argument('-O', '--Imgroot', help='root of Img. End with /', default='./')
    parser.add_argument('-p', '--pool_size', help='num of threads', type=int, default=24)
    return parser


def main():
    parser = argparser()
    args = parser.parse_args()
    # pdb.set_trace()
    Imgroot = args.Imgroot
    newImgroot = args.newImgroot
    newNpyroot = args.newNpyroot
    rf = args.renamefolder
    filelistpath = args.filelistpath
    pool_size = args.pool_size

    file_object = open(filelistpath, 'r')
    try:
        lines = file_object.read()
    except Exception as e:
        raise ('file not read')
    finally:
        file_object.close()

    raw_list = lines.split('\n')
    #    print(raw_list)

    file_list = []
    for raw_line in raw_list:
        if raw_line == '':
            continue
        file_list.append(raw_line)
    # processImage(raw_line,newImgroot,Imgroot,newNpyroot)
    start = time.time()
    # count = 0
    # for raw_line in raw_list:
    #     if raw_line =='':
    #         continue
    #     processImage(raw_line, newImgroot, Imgroot, newNpyroot, rf)
    #     if count%100 == 0 and count>1:
    #         print(count)
    #     count+=1
    #     # file_list.append(raw_line)
    pool = Pool(processes=pool_size)
    process = processor(newImgroot, newNpyroot, Imgroot, rf)
    pool.map(process, file_list)
    pool.close()
    pool.join()
    end = time.time()

    print('%d s used' % (end - start))
    print('%d imgs finished' % len(file_list))


if __name__ == '__main__':
    main()
