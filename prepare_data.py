import os
from shutil import copyfile

voc_dir = '/home/myyu/Downloads/VOCdevkit/VOC2012'

for set_ in ['/train', '/val']:
    if not os.path.isdir('./data' + set_ + '/labels'):
        os.makedirs('./data' + set_ + '/labels')
    if not os.path.isdir('./data' + set_ + '/images'):
        os.makedirs('./data' + set_ + '/images')

    with open(voc_dir + '/ImageSets/Segmentation' + set_ + '.txt') as f:
        content = f.readlines()
        for fn in content:
            print(fn[:-1])
            copyfile(voc_dir + '/JPEGImages/' + fn[:-1] + '.jpg',
                    './data' + set_ + '/images/' + fn[:-1] + '.jpg')
            copyfile(voc_dir + '/SegmentationClass/' + fn[:-1] + '.png',
                    './data' + set_ + '/labels/' + fn[:-1] + '.png')
