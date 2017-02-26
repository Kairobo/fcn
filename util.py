import tensorflow as tf
from glob import glob
from scipy.misc import imread

colors = ((  0,   0,   0), (128,   0,   0), (  0, 128,   0), (128, 128,   0),
            (  0,   0, 128), (128,   0, 128), (  0, 128, 128), (128, 128, 128),
            ( 64,   0,   0), (192,   0,   0), ( 64, 128,   0), (192, 128,   0),
            ( 64,   0, 128), (192,   0, 128), ( 64, 128, 128), (192, 128, 128),
            (  0,  64,   0), (128,  64,   0), (  0, 192,   0), (128, 192,   0),
            (224, 224, 192))

def conv_bn_relu(x, num_filters, ksize=3, stride=1, reuse=None, training=True, name='conv'):
    x = tf.layers.conv2d(x, num_filters, ksize, stride,
                    padding='same', reuse=reuse, use_bias=False,
                    name='%s/conv2d' % name)
    x = tf.layers.batch_normalization(x, reuse=reuse, training=training,
                    beta_initializer=tf.random_normal_initializer(0.1, 0.01),
                    name='%s/bn' % name)
    return tf.nn.relu(x, name='%s/relu' % name)

def upconv_bn_relu(x, num_filters, ksize=4, stride=2, reuse=None, training=True, name='upconv'):
    x = tf.layers.conv2d_transpose(x, num_filters, ksize, stride,
                    padding='same', reuse=reuse, use_bias=False,
                    name='%s/upconv' % name)
    x = tf.layers.batch_normalization(x, reuse=reuse, training=training,
                    beta_initializer=tf.random_normal_initializer(0.1, 0.01),
                    name='%s/bn' % name)
    return tf.nn.relu(x, name='%s/relu' % name)

def load_images(pattern):
    img = []
    for fn in sorted(glob(pattern)):
        img.append(imread(fn, mode='RGB'))

    return img
