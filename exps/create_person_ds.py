import cvbase as cvb
import csv
import lz
from lz import *
import cv2

annop = share_path + 'person_trajectory/' + 'csv/'
imp = share_path + 'person_trajectory/' + 'person/'

annop1 = annop + 'person_location.csv'
f = open(annop1)
t = csv.reader(f)
for row in t:
    # row = t.__iter__().__next__()
    # print(row)
    bbox2 = row[3:7]
    bbox1 = row[7:]
    bbox1 = list(map(int, bbox1))
    bbox2 = list(map(int, bbox2))
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)
    imn = row[1].split('_')[-1] + '.jpg'
    pid = row[2].replace('person', '')

    imp1 = imp + f'person{pid}/' + imn
    x0, y0 = bbox1[0], bbox1[1]


    def shift_bbox(bbox, pnt):
        bbox[0] -= pnt[0]
        bbox[2] -= pnt[0]
        bbox[1] -= pnt[1]
        bbox[3] -= pnt[1]
        return bbox


    bbox1 = shift_bbox(bbox1, (x0, y0))
    bbox2 = shift_bbox(bbox2, (x0, y0))
    if not osp.exists(imp1): continue
    im = cvb.read_img(imp1)
    height, width = bbox1[3] - bbox1[1], bbox1[2] - bbox1[0]
    im = cv2.resize(im, (height, width))
    x, y, x2, y2 = bbox2
    im = im[y:y2, x:x2, :]
    if 0 in im.shape:
        print(im.shape)
        print(row)
    # cvb.show_img(im)
    # cvb.write_img(im, work_path + f'/reid.person/{pid}/{imn}')
    # print(work_path+f'/reid.person/{pid}/{imn}')
