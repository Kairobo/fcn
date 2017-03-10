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
                [  0, 192,   0], [128, 192,   0], [  0,  64, 128],
                [224, 224, 192]], dtype=uint8)
num_classes = colors.shape[0] - 1

def conv_bn_relu(x, num_filters, ksize=3, stride=1, reuse=None, training=True, name='conv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, num_filters, ksize, stride,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d')
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse,
                epsilon=1e-6,scale=False,
                name='bn')
        return tf.nn.relu(x, name='relu')


def upconv_bn_relu(x, num_filters, ksize=3, stride=2, reuse=None, training=True, name='upconv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_filters, ksize, stride,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d_transpose')
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse,
                epsilon=1e-6, scale=False,
                name='bn')
        return tf.nn.relu(x, name='relu')


def load_images(pattern):
    fn = sorted(glob(pattern))
    if 'images' in pattern:
        img = zeros((len(fn), 512, 512, 3), dtype=uint8)
    else:
        img = zeros((len(fn), 512, 512), dtype=uint8)

    for k in range(len(fn)):
        img[k, ...] = imread(fn[k])

    return img


def build_model(x, y, reuse=None, training=True):
    with tf.variable_scope('FCN'):
        # x ~ (?, 512, 512, 3)  RGB images
        # y ~ (?, 512, 512)     Gray scale images with labels as pixel values

        # 512
        conv1 = conv_bn_relu(x, 32, reuse=reuse, training=True, name='conv1/1')
        conv1 = conv_bn_relu(conv1, 32, reuse=reuse, training=True, name='conv1/2')
        conv1 = tf.layers.dropout(conv1, training=training, name='conv1/dropout')

        # 256
        maxpool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool1')
        conv2 = conv_bn_relu(maxpool1, 64, reuse=reuse, training=True, name='conv2/1')
        conv2 = conv_bn_relu(conv2, 64, reuse=reuse, training=True, name='conv2/2')
        conv2 = tf.layers.dropout(conv2, training=training, name='conv2/dropout')

        # 128
        maxpool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool2')
        conv3 = conv_bn_relu(maxpool2, 128, reuse=reuse, training=True, name='conv3/1')
        conv3 = conv_bn_relu(conv3, 128, reuse=reuse, training=True, name='conv3/2')
        conv3 = tf.layers.dropout(conv3, training=training, name='conv3/dropout')

        # 64
        maxpool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='maxpool3')
        conv4 = conv_bn_relu(maxpool3, 256, reuse=reuse, training=True, name='conv4/1')
        conv4 = conv_bn_relu(conv4, 256, reuse=reuse, training=True, name='conv4/2')
        #conv4 = conv_bn_relu(conv4, 512, reuse=reuse, training=training, name='conv4/3')
        conv4 = tf.layers.dropout(conv4, training=training, name='conv4/dropout')

        # 32
        maxpool4 = tf.layers.max_pooling2d(conv4, 2, 2, name='maxpool4')
        conv5 = conv_bn_relu(maxpool4, 512, reuse=reuse, training=True, name='conv5/1')
        conv5 = conv_bn_relu(conv5, 512, reuse=reuse, training=True, name='conv5/2')
        #conv5 = conv_bn_relu(conv5, 512, reuse=reuse, training=True, name='conv5/3')
        conv5 = tf.layers.dropout(conv5, training=training, name='conv5/dropout')

        # 16
        maxpool5 = tf.layers.max_pooling2d(conv5, 2, 2, name='maxpool5')
        conv6 = conv_bn_relu(maxpool5, 1024, reuse=reuse, training=True, name='conv6/1')
        conv6 = conv_bn_relu(conv6, 1024, reuse=reuse, training=True, name='conv6/2')
        #conv6 = conv_bn_relu(conv6, 1024, reuse=reuse, training=True, name='conv6/3')
        conv6 = tf.layers.dropout(conv6, training=training, name='conv6/dropout')

        # 32
        up1 = upconv_bn_relu(conv6, 512, reuse=reuse, training=True, name='up1')
        up1 = tf.concat([up1, conv5], axis=3, name='concat1')

        # 64
        up2 = upconv_bn_relu(up1, 256, reuse=reuse, training=True, name='up2')
        up2 = tf.concat([up2, conv4], axis=3, name='concat2')

        # 128
        up3 = upconv_bn_relu(up2, 128, reuse=reuse, training=True, name='up3')
        up3 = tf.concat([up3, conv3], axis=3, name='concat3')

        # 256
        up4 = upconv_bn_relu(up3, 64, reuse=reuse, training=True, name='up4')
        up4 = tf.concat([up4, conv2], axis=3, name='concat4')

        # 256
        up5 = upconv_bn_relu(up4, 32, reuse=reuse, training=training, name='up5')
        up5 = tf.concat([up5, conv1], axis=3, name='concat5')

        # 512
        logits = tf.layers.conv2d(up5, num_classes, 3,
                                    reuse=reuse, padding='same', name='logits')

        mask = tf.not_equal(y, 255, name='mask')
        logits_masked = tf.boolean_mask(logits, mask)
        labels_masked = tf.boolean_mask(y, mask)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=logits_masked, labels=labels_masked))

    return logits, loss
