from util import *
import tensorflow as tf
import time
from numpy import *
from scipy.misc import imread

def build_model(x, y, reuse=None, training=True):
    with tf.variable_scope('FCN'):
        #conv1_1 = conv_bn_relu(x, 64, reuse=reuse, training=training, name='conv1_1')
        conv1_1 = tf.layers.conv2d(x, 64, 3, 1, padding='same'
                                reuse=reuse, name='conv1_1/conv2d')
        conv1_1 = tf.nn.relu(conv1_1, name='conv1_1/relu')
        conv1_2 = conv_bn_relu(conv1_1, 64, reuse=reuse, training=training, name='conv1_2')
        maxpool1 = tf.layers.max_pooling2d(conv1_2, 2, 2, name='maxpool1') # 256

        conv2_1 = conv_bn_relu(maxpool1, 128, reuse=reuse, training=training, name='conv2_1')
        conv2_2 = conv_bn_relu(conv2_1, 128, reuse=reuse, training=training, name='conv2_2')
        maxpool2 = tf.layers.max_pooling2d(conv2_2, 2, 2, name='maxpool2') # 128

        conv3_1 = conv_bn_relu(maxpool2, 256, reuse=reuse, training=training, name='conv3_1')
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

        logits = tf.layers.conv2d(conv11_1, 20, 3, reuse=reuse, padding='same', name='logits')

        labels = tf.one_hot(y, depth=20, axis=-1, name='one_hot')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                logits=logits, labels=labels))

    return logits, loss

x = tf.placeholder(tf.float32, shape=[None, 512, 512, 3])
y = tf.placeholder(tf.int32, shape=[None, 512, 512])

logits_train, loss_train = build_model(x, y)
logits_val, loss_val = build_model(x, y, reuse=True, training=False)

trainable_vars = tf.trainable_variables()
sum_hist = []
for v in trainable_vars:
    print(v.name, v.get_shape())
    sum_hist.append(tf.summary.histogram(v.name, v))

sum_hist = tf.summary.merge(sum_hist)

sum_loss_train = tf.summary.scalar('loss_train', loss_train)
sum_loss_val = tf.summary.scalar('loss_val', loss_val)

print('\nLoading data ...')
t0 = time.time()
images_train = load_images('./data/train/images/*.png')
labels_train = load_images('./data/train/labels/*.png')

images_val = load_images('./data/val/images/*.png')
labels_val = load_images('./data/val/labels/*.png')
print('Finished loading in %.2f seconds.' % (time.time() - t0))

order = arange(len(images_train))


lr = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_train, var_list=trainable_vars)

batch_size = 2
lr_basic = 1e-4
num_epoch = 10
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    total_count = 0
    t0 = time.time()
    for epoch in range(num_epoch):
        random.shuffle(order)
        for k in range(len(images_train) // batch_size):
            idx = order[(k * batch_size):min((k + 1) * batch_size, 1 + len(images_train))]
            img = images_train[idx, ...]
            lbl = labels_train[idx, ...]

            if random.rand() > 0.5:
                img = img[:, :, ::-1, :]
                lbl = lbl[:, :, ::-1]

            sess.run(train_step, feed_dict={x: img, y: lbl, lr: lr_basic / 2**epoch})

            total_count += 1
            writer.add_summary(sess.run(sum_hist), total_count)

            sum_, l_ = sess.run([sum_loss_train, loss_train], feed_dict={x: img, y: lbl})
            writer.add_summary(sum_, total_count)

            #
            '''
            idx = random.randint(0, len(images_val) - 1)
            img = images_val[idx]
            lbl = labels_val[idx]

            sum_ = sess.run(sum_loss_val, feed_dict={x_raw: img, y_raw: lbl})
            writer.add_summary(sum_, total_count)
            '''
            m, s = divmod(time.time() - t0, 60)
            h, m = divmod(m, 60)
            print('Epoch: [%4d/%4d] [%4d/%4d], Time: [%02d:%02d:%02d], loss: %.4f'
                    % (epoch, num_epoch, k, len(images_train) // batch_size, h, m, s, l_))
