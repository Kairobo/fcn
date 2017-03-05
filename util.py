import tensorflow as tf
from glob import glob
from scipy.misc import imread
from numpy import *

colors = array([[  0,   0,   0], [128,   0,   0], [  0, 128,   0],
                [128, 128,   0], [  0,   0, 128], [128,   0, 128],
                [  0, 128, 128], [128, 128, 128], [ 64,   0,   0],
                [192,   0,   0], [ 64, 128,   0], [192, 128,   0],
                [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128],
                [192, 128, 128], [  0,  64,   0], [128,  64,   0],
                [  0, 192,   0], [128, 192,   0], [224, 224, 192]], dtype=uint8)


def conv_bn(x, num_filters, ksize=4, stride=2, reuse=None, training=True, name='conv', padding='same'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, num_filters, ksize, stride,
                padding=padding, use_bias=False, reuse=reuse,
                name='conv2d')
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse,
                epsilon=1e-6, name='bn')
        return x


def upconv_bn(x, num_filters, ksize=4, stride=2, reuse=None, training=True, name='upconv', padding='same'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_filters, ksize, stride,
                padding=padding, use_bias=False, reuse=reuse,
                name='conv2d_transpose')
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse,
                epsilon=1e-6, name='bn')
        return x


def leakyReLU(x, alpha=0.1, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(alpha * x, x)


def load_images(pattern):
    fn = sorted(glob(pattern))#[:50]
    if 'images' in pattern:
        img = zeros((len(fn), 512, 512, 3), dtype=uint8)
    else:
        img = zeros((len(fn), 512, 512), dtype=uint8)

    for k in range(len(fn)):
        img[k, ...] = imread(fn[k])

    return img


def build_model(x, y, num_filters=64, reuse=None, training=True):
    with tf.variable_scope('FCN'):
        e1 = tf.layers.conv2d(x, num_filters, 4, 2,
                padding='same', reuse=reuse,
                name='e1/conv2d')
        e1 = tf.layers.dropout(e1, training=training, name='e1/dropout')

        e2 = conv_bn(tf.nn.elu(e1, name='e2/elu'), 2 * num_filters,
                reuse=reuse, training=training, name='e2')
        e2 = tf.layers.dropout(e2, training=training, name='e2/dropout')

        e3 = conv_bn(tf.nn.elu(e2, name='e3/elu'), 4 * num_filters,
                reuse=reuse, training=training, name='e3')
        e3 = tf.layers.dropout(e3, training=training, name='e3/dropout')

        e4 = conv_bn(tf.nn.elu(e3, name='e4/elu'), 8 * num_filters,
                reuse=reuse, training=training, name='e4')
        e5 = conv_bn(tf.nn.elu(e4, name='e5/elu'), 8 * num_filters,
                reuse=reuse, training=training, name='e5')
        e6 = conv_bn(tf.nn.elu(e5, name='e6/elu'), 8 * num_filters,
                reuse=reuse, training=training, name='e6')
        e7 = conv_bn(tf.nn.elu(e6, name='e7/elu'), 8 * num_filters,
                reuse=reuse, training=training, name='e7')

        d1 = upconv_bn(tf.nn.elu(e7, name='d1/elu'), 8 * num_filters,
                reuse=reuse, training=training, name='d1')
        d1 = tf.concat([d1, e6], axis=-1, name='d1/concat')

        d2 = upconv_bn(tf.nn.elu(d1, name='d2/elu'), 8 * num_filters,
                reuse=reuse, training=training, name='d2')
        d2 = tf.concat([d2, e5], axis=-1, name='d2/concat')

        d3 = upconv_bn(tf.nn.elu(d2, name='d3/elu'), 8 * num_filters,
                reuse=reuse, training=training, name='d3')
        d3 = tf.concat([d3, e4], axis=-1, name='d3/concat')

        d4 = upconv_bn(tf.nn.elu(d3, name='d4/elu'), 4 * num_filters,
                reuse=reuse, training=training, name='d4')
        d4 = tf.concat([d4, e3], axis=-1, name='d4/concat')

        d5 = upconv_bn(tf.nn.elu(d4, name='d5/elu'), 2 * num_filters,
                reuse=reuse, training=training, name='d5')
        d5 = tf.concat([d5, e2], axis=-1, name='d5/concat')

        d6 = upconv_bn(tf.nn.elu(d5, name='d6/elu'), num_filters,
                reuse=reuse, training=training, name='d6')
        d6 = tf.concat([d6, e1], axis=-1, name='d6/concat')

        logits = tf.layers.conv2d_transpose(tf.nn.elu(d6, name='d7/elu'),
                20, 4, 2, padding='same', reuse=reuse,
                name='logits')

        labels = tf.one_hot(y, depth=20, axis=-1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                logits=logits, labels=labels))

    return logits, loss

def VGG16_FCN_8s(x, y, num_filters=64, reuse=None, training=True):

    with tf.variable_scope('VGG16_FCN'):

        conv1_1 = tf.layers.conv2d(x, num_filters, 3, 1,
                padding='same', reuse=reuse,
                name='conv1_1/conv2d')

        conv1_2 = conv_bn(tf.nn.elu(conv1_1, name='conv1_2/elu'), num_filters, 3, 1,
                reuse=reuse, training=training, name='conv1_2')

        pool1 = tf.layers.max_pooling2d(tf.nn.elu(conv1_2, name='pool1/elu'), 
                [1, 2, 2, 1], [1, 2, 2, 1], 
                padding='same', 
                name='pool1')

        conv2_1 = conv_bn(tf.nn.elu(pool1, name='conv2_1/elu'), 2*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv2_1')

        conv2_2 = conv_bn(tf.nn.elu(conv2_1, name='conv2_2/elu'), 2*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv2_2')

        pool2 = tf.layers.max_pooling2d(tf.nn.elu(conv2_2, name='pool2/elu'), 
                [1, 2, 2, 1], [1, 2, 2, 1], 
                padding='same', 
                name='pool2')

        conv3_1 = conv_bn(tf.nn.elu(pool2, name='conv3_1/elu'), 4*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv3_1')

        conv3_2 = conv_bn(tf.nn.elu(conv3_1, name='conv3_2/elu'), 4*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv3_2')

        conv3_3 = conv_bn(tf.nn.elu(conv3_2, name='conv3_3/elu'), 4*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv3_3')

        pool3 = tf.layers.max_pooling2d(tf.nn.elu(conv3_3, name='pool3/elu'), 
                [1, 2, 2, 1], [1, 2, 2, 1], 
                padding='same', 
                name='pool3')

        conv4_1 = conv_bn(tf.nn.elu(pool3, name='conv4_1/elu'), 8*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv4_1')

        conv4_2 = conv_bn(tf.nn.elu(conv4_1, name='conv4_2/elu'), 8*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv4_2')

        conv4_3 = conv_bn(tf.nn.elu(conv4_2, name='conv4_3/elu'), 8*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv4_3')

        pool4 = tf.layers.max_pooling2d(tf.nn.elu(conv4_3, name='pool4/elu'), 
                [1, 2, 2, 1], [1, 2, 2, 1], 
                padding='same', 
                name='pool4')

        conv5_1 = conv_bn(tf.nn.elu(pool4, name='conv5_1/elu'), 8*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv5_1')

        conv5_2 = conv_bn(tf.nn.elu(conv5_1, name='conv5_2/elu'), 8*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv5_2')

        conv5_3 = conv_bn(tf.nn.elu(conv5_2, name='conv5_3/elu'), 8*num_filters, 3, 1,
                reuse=reuse, training=training, name='conv5_3')

        pool5 = tf.layers.max_pooling2d(tf.nn.elu(conv5_3, name='pool5/elu'), 
                [1, 2, 2, 1], [1, 2, 2, 1], 
                padding='same', 
                name='pool5')
        #FCN 
        conv6_1 = conv_bn(tf.nn.elu(pool5, name='conv6_1/elu'), 64*num_filters, 7, 1,
                reuse=reuse, training=training, name='conv6_1', padding='valid')

        conv6_2 = conv_bn(tf.nn.elu(conv6_1, name='conv6_2/elu'), 64*num_filters, 1, 1,
                reuse=reuse, training=training, name='conv6_2', padding='valid')
        conv6_2 = tf.layers.dropout(conv6_2, training=training, name='conv6_2/dropout')

        conv6_3 = conv_bn(tf.nn.elu(conv6_2, name='conv6_3/elu'), 1000, 1, 1,
                reuse=reuse, training=training, name='conv6_3', padding='valid')
        conv6_3 = tf.layers.dropout(conv6_3, training=training, name='conv6_3/dropout')
        
        #Upsample
        up_1 = upconv_bn(tf.nn.elu(conv6_3, name='up_1/elu'), 64*num_filters, 7, 1
                reuse=reuse, training=training, name='up_1', padding='valid')

        up_2 = upconv_bn(tf.nn.elu(d1_1, name='up_2/elu'), 4*num_filters, 8, 4
                reuse=reuse, training=training, name='up_2', padding='same')

        pool4_up = upconv_bn(tf.nn.elu(pool4, name='pool4_up/elu'), 4*num_filters, 4, 2
                reuse=reuse, training=training, name='pool4_up', padding='same')

        up_3 = tf.add(pool3, pool4_up)
        up_3 = tf.add(up_3, up_2)

        logits = upconv_bn(tf.nn.elu(up_3, name='logits/elu'), 20, 16, 8
                reuse=reuse, training=training, name='logits', padding='same')

        labels = tf.one_hot(y, depth=20, axis=-1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                logits=logits, labels=labels))  

      




