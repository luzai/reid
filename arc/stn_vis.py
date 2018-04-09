# encoding: utf-8

import os
import glob
import torch
import random
import argparse
import numpy as np
from grid_sample import grid_sample
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
from exps.train import *

lz.init_dev((3,))
args = torchpack.load_cfg('conf/trihard.py')
args = args.cfgs[0]

dataset, num_classes, train_loader, val_loader, test_loader = \
    get_data(args.dataset, args.split, args.data_dir, args.height,
             args.width, args.batch_size, args.num_instances, args.workers,
             args.combine_trainval, pin_memory=args.pin_mem, name_val=args.dataset_val)
train_loader.shuffle = True
train_loader.sampler = None

model = models.create(args.arch,
                      pretrained=args.pretrained,
                      dropout=args.dropout,
                      norm=args.normalize,
                      num_features=args.global_dim,
                      num_classes=args.num_classes
                      )

model = nn.DataParallel(model).cuda()

image_dir = 'work/imgs.3/'
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)

target2data_list = collections.defaultdict(list)
total = 0
N = 10
for data_batch, _, target_batch, _ in train_loader:
    for data, target in zip(data_batch, target_batch):
        data_list = target2data_list[target]
        if len(data_list) < N:
            data_list.append(data)
            total += 1
    if total == N * 10:
        break
data_list = [target2data_list[i][j] for i in target2data_list.keys() for j in range(len(target2data_list[i]))]
source_data = torch.stack(data_list)
source_data = source_data.cuda()

batch_size = N * 10
frames_list = [[] for _ in range(batch_size)]
from torchvision import  utils as vutils
paths = sorted(glob.glob('work/tps.3/checkpoint.*.pth'))[::-1]
for pi, path in enumerate(paths):  # path index
    # path = 'work/tps.3/checkpoint.{}.pth'.format(pi)
    print('path %d/%d: %s' % (pi, len(paths), path))
    # model.load_state_dict(torch.load(path)['state_dict'])
    load_state_dict(model, torch.load(path)['state_dict'])
    source_control_points = model.module.stn.loc_net(Variable(source_data, volatile=True))
    source_coordinate = model.module.stn.tps(source_control_points)
    grid = source_coordinate.view(batch_size, 256, 128, 2)
    target_data = grid_sample(Variable(source_data, volatile=True), grid).data

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min)


    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, t.min(), t.max())

    for s in source_data:
        norm_range(s,None)
    for t in target_data:
        norm_range(t,None)

    source_array = (source_data  * 255).cpu().numpy().astype('uint8')
    target_array = (target_data  * 255).cpu().numpy().astype('uint8')
    source_array = source_array.transpose((0,2,3,1))
    target_array=target_array.transpose((0,2,3,1))
    for si in range(batch_size):  # sample index
        # resize for better visualization
        source_image = Image.fromarray(source_array[si]).convert('RGB')
        target_image = Image.fromarray(target_array[si]).convert('RGB')
        # create grey canvas for external control points
        canvas = Image.new(mode='RGB', size=(64 * 7, 64 * 5), color=(255,255,255))
        canvas.paste(source_image, (64, 64))
        canvas.paste(target_image, (64 * 4, 64))
        source_points = source_control_points.data[si]
        source_points[:,0] = (source_points[:,0] + 1) / 2 * 128 + 64
        source_points[:,1] = (source_points[:,1] + 1) / 2 * 256 + 64

        draw = ImageDraw.Draw(canvas)
        for x, y in source_points:
            draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))
        source_points = source_points.view(8, 4, 2)
        for j in range(8):
            for k in range(4):
                x1, y1 = source_points[j, k]
                if j > 0:  # connect to left
                    x2, y2 = source_points[j - 1, k]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
                if k > 0:  # connect to up
                    x2, y2 = source_points[j, k - 1]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
        draw.text((10, 0), 'sample %03d, iter %03d' % (si, (len(paths) - 1 - pi)), fill=(255, 0, 0), )
        canvas.save(image_dir + 'sample%03d_iter%03d.png' % (si, len(paths) - 1 - pi))
        # exit(-1)