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

vgg_weights = load('vgg16.npy', encoding='latin1').item()

def conv_bn_relu_vgg(x, reuse=None, name='conv_vgg'):
    if 'conv' in name:
        kernel = vgg_weights[name][0]
    elif name == 'fc6':
        kernel = vgg_weights[name][0].reshape([7, 7, 512, 4096])

    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[0],
                padding='same', use_bias=False, reuse=reuse,
                kernel_initializer=tf.constant_initializer(kernel),
                name='conv2d')
        x = tf.layers.batch_normalization(x, training=True, reuse=reuse,
                epsilon=1e-6, scale=False,
                name='bn')
        return tf.nn.relu(x, name='relu')


def conv_bn_relu(x, num_filters, reuse=None, name='conv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, num_filters, 3,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d')
        x = tf.layers.batch_normalization(x, training=True, reuse=reuse,
                epsilon=1e-6, scale=False,
                name='bn')
        return tf.nn.relu(x, name='relu')


def upconv_bn_relu(x, num_filters, ksize=3, stride=2, reuse=None, name='upconv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_filters, ksize, stride,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d_transpose')
        x = tf.layers.batch_normalization(x, training=True, reuse=reuse,
                epsilon=1e-6, #scale=False,
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

        # VGG takes BGR images
        x = x[..., ::-1] - [103.939, 116.779, 123.68]

        # 512
        conv1 = conv_bn_relu_vgg(x, reuse=reuse, name='conv1_1')
        conv1 = conv_bn_relu_vgg(conv1, reuse=reuse, name='conv1_2')

        # 256
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')
        conv2 = conv_bn_relu_vgg(pool1, reuse=reuse, name='conv2_1')
        conv2 = conv_bn_relu_vgg(conv2, reuse=reuse, name='conv2_2')

        # 128
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')
        conv3 = conv_bn_relu_vgg(pool2, reuse=reuse, name='conv3_1')
        conv3 = conv_bn_relu_vgg(conv3, reuse=reuse, name='conv3_2')
        conv3 = conv_bn_relu_vgg(conv3, reuse=reuse, name='conv3_3')

        # 64
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='pool3')
        conv4 = conv_bn_relu_vgg(pool3, reuse=reuse, name='conv4_1')
        conv4 = conv_bn_relu_vgg(conv4, reuse=reuse, name='conv4_2')
        conv4 = conv_bn_relu_vgg(conv4, reuse=reuse, name='conv4_3')

        # 32
        pool4 = tf.layers.max_pooling2d(conv4, 2, 2, name='pool4')
        conv5 = conv_bn_relu_vgg(pool4, reuse=reuse, name='conv5_1')
        conv5 = conv_bn_relu_vgg(conv5, reuse=reuse, name='conv5_2')
        conv5 = conv_bn_relu_vgg(conv5, reuse=reuse, name='conv5_3')

        # 16
        pool5 = tf.layers.max_pooling2d(conv5, 2, 2, name='pool5')
        #conv6 = tf.layers.dropout(pool5, training=training, name='dropout6_1')
        conv6 = conv_bn_relu(pool5, 512, reuse=reuse, name='conv6_1')
        #conv6 = tf.layers.dropout(conv6, training=training, name='dropout6_2')
        conv6 = conv_bn_relu(conv6, 512, reuse=reuse, name='conv6_2')
        #conv6 = tf.layers.dropout(conv6, training=training, name='dropout6_3')
        conv6 = conv_bn_relu(conv6, 512, reuse=reuse, name='conv6_3')

        # 32
        up1 = upconv_bn_relu(conv6, 512, reuse=reuse, name='up1')
        #up1 = tf.concat([up1, conv5], axis=3, name='concat1')
        up1 = tf.add(up1, conv5, name='add1')

        # 64
        up2 = upconv_bn_relu(up1, 512, reuse=reuse, name='up2')
        #up2 = tf.concat([up2, conv4], axis=3, name='concat2')
        up2 = tf.add(up2, conv4, name='add2')

        # 128
        up3 = upconv_bn_relu(up2, 256, reuse=reuse, name='up3')
        #up3 = tf.concat([up3, conv3], axis=3, name='concat3')
        up3 = tf.add(up3, conv3, name='add3')

        # 256
        up4 = upconv_bn_relu(up3, 128, reuse=reuse, name='up4')
        #up4 = tf.concat([up4, conv2], axis=3, name='concat4')
        up4 = tf.add(up4, conv2, name='add4')

        # 256
        up5 = upconv_bn_relu(up4, 64, reuse=reuse, name='up5')
        #up5 = tf.concat([up5, conv1], axis=3, name='concat5')
        up5 = tf.add(up5, conv1, name='add5')

        # 512
        with tf.variable_scope('logits'):
            logits = tf.layers.conv2d(up5, num_classes, 3, padding='same',
                        use_bias=False, reuse=reuse, name='conv2d')
            logits = tf.layers.batch_normalization(logits,
                        training=True, reuse=reuse, epsilon=1e-6, name='bn')

        mask = tf.not_equal(y, 255, name='mask')
        logits_masked = tf.boolean_mask(logits, mask)
        labels_masked = tf.boolean_mask(y, mask)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=logits_masked, labels=labels_masked))

    return logits, loss
