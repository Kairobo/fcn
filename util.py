import tensorflow as tf
from glob import glob
from scipy.misc import imread
from numpy import *

colors = array([[  0,   0,   0], [128,   0,   0], [  0, 128,   0], [128, 128,   0],
                [  0,   0, 128], [128,   0, 128], [  0, 128, 128], [128, 128, 128],
                [ 64,   0,   0], [192,   0,   0], [ 64, 128,   0], [192, 128,   0],
                [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128], [192, 128, 128],
                [  0,  64,   0], [128,  64,   0], [  0, 192,   0], [128, 192,   0],
                [224, 224, 192]], dtype=uint8)

def conv_bn_relu(x, num_filters, ksize=3, stride=1, reuse=None, training=True, name='conv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, num_filters, ksize, stride,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d')
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse,
                epsilon=1e-6, name='bn')
        return tf.nn.relu(x, name='relu')

def upconv_bn_relu(x, num_filters, ksize=4, stride=2, reuse=None, training=True, name='upconv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_filters, ksize, stride,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d_transpose')
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse,
                epsilon=1e-6, name='bn')
        return tf.nn.relu(x, name='relu')

def load_images(pattern):
    fn = sorted(glob(pattern))#[:50]
    if 'images' in pattern:
        img = zeros((len(fn), 512, 512, 3), dtype=uint8)
    else:
        img = zeros((len(fn), 512, 512), dtype=uint8)

    for k in range(len(fn)):
        img[k, ...] = imread(fn[k])

    return img
