from glob import glob
from scipy.misc import imread
from math import ceil
import tensorflow as tf
import numpy as np 

VGG_MEAN_RGB = [123.68, 116.779, 103.939]

colors = np.array([[  0,   0,   0], [128,   0,   0], [  0, 128,   0],
                [128, 128,   0], [  0,   0, 128], [128,   0, 128],
                [  0, 128, 128], [128, 128, 128], [ 64,   0,   0],
                [192,   0,   0], [ 64, 128,   0], [192, 128,   0],
                [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128],
                [192, 128, 128], [  0,  64,   0], [128,  64,   0],
                [  0, 192,   0], [128, 192,   0], [224, 224, 192]], dtype=np.uint8)

def load_images(pattern):
    fn = sorted(glob(pattern))#[:50]
    if 'images' in pattern:
        img = np.zeros((len(fn), 512, 512, 3), dtype=np.uint8)
    else:
        img = np.zeros((len(fn), 512, 512), dtype=np.uint8)
    for k in range(len(fn)):
        img[k, ...] = imread(fn[k])
    return img

def build_model(x, y, weights_path, num_classes=21):
    weights = np.load(weights_path, encoding='latin1').item()
    print('Pretrained weights loaded')

    # define some helper functions
    def conv_layer(bottom, name):
        with tf.variable_scope(name) as scope:
            init = tf.constant_initializer(value=weights[name][0], dtype=tf.float32)
            shape = weights[name][0].shape
            print('Layer name: %s' % name)
            print('Layer shape: %s' % str(shape))
            kernel = tf.get_variable(name='filter', initializer=init, shape=shape)
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
            bias_weights = weights[name][1]
            shape = weights[name][1].shape
            init = tf.constant_initializer(value=bias_weights, dtype=tf.float32)
            conv_biases = tf.get_variable(name='biases', initializer=init, shape=shape)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(bottom, name):
        with tf.variable_scope(name) as scope:
            if name == 'fc6':
                shape = [7, 7, 512, 4096]
                print('Layer name: %s' % name)
                print('Layer shape: %s' % shape)
                kernel = weights[name][0]
                kernel = kernel.reshape(shape)
                init = tf.constant_initializer(value=kernel, dtype=tf.float32)
                kernel = tf.get_variable(name='weights', initializer=init, shape=shape)
            else:
                shape = [1, 1, 4096, 4096]
                print('Layer name: %s' % name)
                print('Layer shape: %s' % shape)
                kernel = weights[name][0]
                kernel = kernel.reshape(shape)
                init = tf.constant_initializer(value=kernel, dtype=tf.float32)
                kernel = tf.get_variable(name='weights', initializer=init, shape=shape)
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
            bias_weights = weights[name][1]
            shape = weights[name][1].shape
            init = tf.constant_initializer(value=bias_weights, dtype=tf.float32)
            conv_biases = tf.get_variable(name='biases', initializer=init, shape=shape)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return bias

    def score_layer(bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            num_in_channels = bottom.get_shape()[3].value
            shape = [1, 1, num_in_channels, num_classes]
            stddev = (2/num_in_channels)**0.5
            init = tf.truncated_normal_initializer(stddev=stddev)
            kernel = tf.get_variable('weights', shape=shape, initializer=init)
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
            init = tf.constant_initializer(0.0)
            conv_biases = tf.get_variable(name='biases', shape=[num_classes], initializer=init)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias

    def upscore_layer(bottom, shape, num_classes, name, ksize=4, stride=2):
        with tf.variable_scope(name) as scope:
            num_in_channels = bottom.get_shape()[3].value
            output_shape = tf.stack([shape[0], shape[1], shape[2], num_classes])
            f_shape = [ksize, ksize, num_classes, num_in_channels]

            num_input = ksize*ksize*num_in_channels/stride
            stddev = (2/num_input)**0.5

            width = f_shape[0]
            height = f_shape[0]
            f = ceil(width/2.0)
            c = (2*f-1-f%2)/(2.0*f)
            bilinear = np.zeros([f_shape[0], f_shape[1]])
            for x in range(width):
                for y in range(height):
                    value = (1-abs(x/f-c))*(1-abs(y/f-c))
                    bilinear[x,y] = value
            kernel = np.zeros(f_shape)
            for i in range(f_shape[2]):
                kernel[:, :, i, i] = bilinear
            init = tf.constant_initializer(value=kernel, dtype=tf.float32)
            filt = tf.get_variable(name='up_filter', initializer=init, shape=kernel.shape)
            deconv = tf.nn.conv2d_transpose(bottom, filt, output_shape, strides=[1, stride, stride, 1], padding='SAME')
            return deconv

    with tf.name_scope('preprocess') as scope:
        r, g, b = tf.split(x, 3, 3)

        bgr = tf.concat([b-VGG_MEAN_RGB[2], g-VGG_MEAN_RGB[1], r-VGG_MEAN_RGB[0]],3)

    conv1_1 = conv_layer(bgr,"conv1_1")
    conv1_2 = conv_layer(conv1_1, "conv1_2")
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

    conv2_1 = conv_layer(pool1,"conv2_1")
    conv2_2 = conv_layer(conv2_1, "conv2_2")
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")

    conv3_1 = conv_layer(pool2,"conv3_1")
    conv3_2 = conv_layer(conv3_1, "conv3_2")
    conv3_3 = conv_layer(conv3_2, "conv3_3")
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")

    conv4_1 = conv_layer(pool3,"conv4_1")
    conv4_2 = conv_layer(conv4_1, "conv4_2")
    conv4_3 = conv_layer(conv4_2, "conv4_3")
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")

    conv5_1 = conv_layer(pool4,"conv5_1")
    conv5_2 = conv_layer(conv5_1, "conv5_2")
    conv5_3 = conv_layer(conv5_2, "conv5_3")
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool5")

    fc6 =  fc_layer(pool5, "fc6")
    fc7 = fc_layer(fc6, "fc7")
    score_fr = score_layer(fc7, "score_fr", num_classes)
    upscore = upscore_layer(score_fr, shape=tf.shape(bgr), num_classes=num_classes, name="up", ksize=64, stride=32)

    mask = tf.not_equal(y, 255, name='mask')
    logits_masked = tf.boolean_mask(upscore, mask)
    labels_masked = tf.boolean_mask(y, mask)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_masked, labels=labels_masked))

    return upscore, loss
