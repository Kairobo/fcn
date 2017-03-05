#!/usr/bin/env python
from PIL import Image	# required to read indexed PNG files used in VOC labels
import numpy as np 
import tensorflow as tf
import os

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

rootdir = '/mnt/ngv/datasets/VOC/VOCdevkit/VOC2012'
imagesets_file = os.path.join(rootdir, 'ImageSets','Segmentation','train.txt')
filename_pairs = []
with open(imagesets_file,'r') as f:
	lines = f.readlines()
	for line in lines:
		img_path = os.path.join(rootdir,'JPEGImages',line.strip() + ".jpg")
		annotation_path = os.path.join(rootdir,'SegmentationClass',line.strip() + ".png")
		filename_pairs.append((img_path,annotation_path))

tfrecords_filename = './data/pascal_voc_segmentation.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for img_path, annotation_path in filename_pairs:
	print("Processing",img_path)
	img = np.array(Image.open(img_path))
	annotation = np.array(Image.open(annotation_path))

	height = img.shape[0]
	width = img.shape[1]

	img_raw = img.tostring()
	annotation_raw = annotation.tostring()

	example = tf.train.Example(features=tf.train.Features(feature={
		'height': _int64_feature(height),
		'width': _int64_feature(width),
		'image_raw': _bytes_feature(img_raw),
		'mask_raw': _bytes_feature(annotation_raw)}))
	writer.write(example.SerializeToString())
writer.close()