# encoding: utf-8

import os
import glob
import imageio
import argparse

gif_dir = 'work/gifs.3/'
if not os.path.isdir(gif_dir):
    os.makedirs(gif_dir)

max_iter = 100
for i in range(max_iter):
    print('sample %d' % i)
    paths = sorted(glob.glob('work/imgs.3/sample%03d_*.png' % (
        i,
    )))
    paths = [path for i, path in enumerate(paths) if i % 2 == 0]
    images = [imageio.imread(path) for path in paths]
    for _ in range(10): images.append(images[-1])  # delay at the end
    imageio.mimsave(gif_dir + 'sample%03d.gif' % i, images)
