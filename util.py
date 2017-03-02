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


def build_model(x, y, reuse=None, training=True):
    with tf.variable_scope('FCN'):
        conv1_1 = tf.layers.conv2d(x, 64, 5, 2, padding='same',
                                reuse=reuse, name='conv1_1/conv2d')
        conv1_1 = tf.nn.relu(conv1_1, name='conv1_1/relu')
        #conv1_1 = tf.layers.dropout(conv1_1, training=training, name='conv1_1/dropout')
        conv1_2 = conv_bn_relu(conv1_1, 64, reuse=reuse, training=training, name='conv1_2')
        maxpool1 = tf.layers.max_pooling2d(conv1_2, 2, 2, name='maxpool1') # 256

        conv2_1 = conv_bn_relu(maxpool1, 128, reuse=reuse, training=training, name='conv2_1')
        #conv2_1 = tf.layers.dropout(conv2_1, training=training, name='conv2_1/dropout')
        conv2_2 = conv_bn_relu(conv2_1, 128, reuse=reuse, training=training, name='conv2_2')
        maxpool2 = tf.layers.max_pooling2d(conv2_2, 2, 2, name='maxpool2') # 128

        conv3_1 = conv_bn_relu(maxpool2, 256, reuse=reuse, training=training, name='conv3_1')
        #conv3_1 = tf.layers.dropout(conv3_1, training=training, name='conv3_1/dropout')
        conv3_2 = conv_bn_relu(conv3_1, 256, reuse=reuse, training=training, name='conv3_2')
        maxpool3 = tf.layers.max_pooling2d(conv3_2, 2, 2, name='maxpool3') # 64

        conv4_1 = conv_bn_relu(maxpool3, 512, reuse=reuse, training=training, name='conv4_1')
        conv4_2 = conv_bn_relu(conv4_1, 512, reuse=reuse, training=training, name='conv4_2')
        maxpool4 = tf.layers.max_pooling2d(conv4_2, 2, 2, name='maxpool4') # 32

        conv5_1 = conv_bn_relu(maxpool4, 512, reuse=reuse, training=training, name='conv5_1')
        conv5_2 = conv_bn_relu(conv5_1, 512, reuse=reuse, training=training, name='conv5_2')
        maxpool5 = tf.layers.max_pooling2d(conv5_2, 2, 2, name='maxpool5') # 16

        conv6_1 = conv_bn_relu(maxpool5, 512, reuse=reuse, training=training, name='conv6_1')
        conv6_2 = conv_bn_relu(conv6_1, 512, reuse=reuse, training=training, name='conv6_2')

        up1 = upconv_bn_relu(conv6_2, 512, reuse=reuse, training=training, name='up1')
        up1 = tf.concat([up1, conv5_2], axis=3, name='concat1')
        conv7_1 = conv_bn_relu(up1, 512, reuse=reuse, training=training, name='conv7_1')

        up2 = upconv_bn_relu(conv7_1, 512, reuse=reuse, training=training, name='up2')
        up2 = tf.concat([up2, conv4_2], axis=3, name='concat2')
        conv8_1 = conv_bn_relu(up2, 512, reuse=reuse, training=training, name='conv8_1')

        up3 = upconv_bn_relu(conv8_1, 256, reuse=reuse, training=training, name='up3')
        up3 = tf.concat([up3, conv3_2], axis=3, name='concat3')
        conv9_1 = conv_bn_relu(up3, 256, reuse=reuse, training=training, name='conv9_1')

        up4 = upconv_bn_relu(conv9_1, 128, reuse=reuse, training=training, name='up4')
        up4 = tf.concat([up4, conv2_2], axis=3, name='concat4')
        conv10_1 = conv_bn_relu(up4, 128, reuse=reuse, training=training, name='conv10_1')

        up5 = upconv_bn_relu(conv10_1, 64, reuse=reuse, training=training, name='up5')
        up5 = tf.concat([up5, conv1_2], axis=3, name='concat5')
        conv11_1 = conv_bn_relu(up5, 64, reuse=reuse, training=training, name='conv11_1')

        logits = tf.layers.conv2d_transpose(conv11_1, 20, 4, 2,
                                reuse=reuse, padding='same', name='logits')

        labels = tf.one_hot(y, depth=20, axis=-1, name='one_hot')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                logits=logits, labels=labels))

    return logits, loss
