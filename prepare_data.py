import os
import sys
from scipy.misc import imread, imsave
from numpy import *
from util import colors

voc_dir = '/home/myyu/Downloads/VOCdevkit/VOC2012'
set_ = sys.argv[1] # either being 'train' or 'val'

if not os.path.isdir('./data/' + set_ + '/labels'):
    os.makedirs('./data/' + set_ + '/labels')
if not os.path.isdir('./data/' + set_ + '/images'):
    os.makedirs('./data/' + set_ + '/images')

with open(voc_dir + '/ImageSets/Segmentation/' + set_ + '.txt') as f:
    content = f.readlines()
    for line in content:
        fn = line[:-1]
        print('%s/%s' % (set_, fn))
        img = imread(voc_dir + '/JPEGImages/' + fn + '.jpg', mode='RGB')
        data = zeros((512, 512, 3), dtype=uint8)
        data[:img.shape[0], :img.shape[1], :] = img
        a0, b0 = divmod(img.shape[0], 256)
        data[-2, -2, :] = a0
        data[-2, -1, :] = b0

        a1, b1 = divmod(img.shape[1], 256)
        data[-1, -2, :] = a1
        data[-1, -1, :] = b1
        imsave('./data/' + set_ + '/images/' + fn + '.png', data)

        lbl = imread(voc_dir + '/SegmentationClass/' + fn + '.png', mode='RGB')
        temp = zeros((lbl.shape[0], lbl.shape[1]), dtype=uint8) + 255

        for k in range(20):
            clr = int32(colors[k, :])
            e = lbl - clr[newaxis, newaxis, :]
            temp[sum(e**2, axis=2) == 0] = k

        data = zeros((512, 512), dtype=uint8) + 255
        data[:lbl.shape[0], :lbl.shape[1]] = temp

        imsave('./data/' + set_ + '/labels/' + fn + '.png', data)


if not os.path.isdir('./logs'):
    os.makedirs('./logs')

if not os.path.isdir('./checkpoint'):
    os.makedirs('./checkpoint')
