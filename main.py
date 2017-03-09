from util import *
import tensorflow as tf
import time, os
from numpy import *
from scipy.misc import imsave

tf.app.flags.DEFINE_integer('batch_size', 4, 'Number of images in each batch')
tf.app.flags.DEFINE_integer('num_epoch', 100, 'Total number of epochs to run for training')
tf.app.flags.DEFINE_boolean('training', True, 'If true, train the model; otherwise evaluate the existing model')
tf.app.flags.DEFINE_float('basic_learning_rate', 1e-4, 'Initial learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_ratio', 0.99, 'Ratio for decaying the learning rate after every epoch')
tf.app.flags.DEFINE_float('min_learning_rate', 1e-6, 'Minimum learning rate used for training')
tf.app.flags.DEFINE_string('gpu', '0', 'GPU to be used')

config = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

x = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='x')
y = tf.placeholder(tf.int64, shape=[None, 512, 512], name='y')

logits, loss = build_model(x, y)
logits_val, loss_val = build_model(x, y, training=False, reuse=True)

vars_trainable = tf.trainable_variables()
for var in vars_trainable:
    #print(var.name, var.get_shape())
    tf.summary.histogram(var.name, var)

tf.summary.scalar('loss', loss)

pred_train = tf.argmax(logits, axis=-1)
result_train = tf.concat([y, pred_train], axis=2)
result_train = tf.cast(12 * tf.reshape(result_train, [-1, 512, 1024, 1]), tf.uint8)

pred_val = tf.argmax(logits_val, axis=-1)
result_val = tf.concat([y, pred_val], axis=2)
result_val = tf.cast(12 * tf.reshape(result_val, [-1, 512, 1024, 1]), tf.uint8)

tf.summary.image('result_train', result_train, max_outputs=config.batch_size)
tf.summary.image('result_val', result_val, max_outputs=config.batch_size)

learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
tf.summary.scalar('learning_rate', learning_rate)

sum_all = tf.summary.merge_all()

####
print('\nLoading data ...')
t0 = time.time()
if config.training:
    images = load_images('./data/train/images/*.png')
    labels = load_images('./data/train/labels/*.png')
else:
    images = load_images('./data/val/images/*.png')
    labels = load_images('./data/val/labels/*.png')
print('Finished loading in %.2f seconds.' % (time.time() - t0))

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=vars_trainable)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=10)
    ckpt = tf.train.get_checkpoint_state('./checkpoint')
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join('./checkpoint', ckpt_name))
        print('[*] Success to read {}'.format(ckpt_name))
    else:
        if config.training:
            print('[*] Failed to find a checkpoint. Start training from scratch ...')
        else:
            raise ValueError('[*] Failed to find a checkpoint.')

    if config.training:
        writer = tf.summary.FileWriter("./logs", sess.graph)

        order = arange(images.shape[0], dtype=uint32)
        total_count = 0
        t0 = time.time()
        for epoch in range(config.num_epoch):
            random.shuffle(order)
            lr = max(config.basic_learning_rate * config.learning_rate_decay_ratio**epoch, config.min_learning_rate)
            for k in range(images.shape[0] // config.batch_size):
                idx = order[(k * config.batch_size):min((k + 1) * config.batch_size, 1 + images.shape[0])]
                if random.rand() > 0.5:
                    img = images[idx, :, :, :]
                    lbl = labels[idx, :, :]
                else:
                    img = images[idx, :, ::-1, :]
                    lbl = labels[idx, :, ::-1]

                l_train, _ = sess.run([loss, train_step], feed_dict={x: img, y: lbl, learning_rate: lr})

                if total_count % (images.shape[0] // config.batch_size // 10) == 0:
                    writer.add_summary(sess.run(sum_all, feed_dict={x: img, y: lbl, learning_rate: lr}), total_count)
                total_count += 1

                m, s = divmod(time.time() - t0, 60)
                h, m = divmod(m, 60)
                print('Epoch: [%4d/%4d] [%4d/%4d], Time: [%02d:%02d:%02d], loss: %.4f'
                        % (epoch, config.num_epoch, k, images.shape[0] // config.batch_size, h, m, s, l_train))

            if epoch % 10 == 0:
                print('Saving checkpoint ...')
                saver.save(sess, './checkpoint/FCN.ckpt', global_step=epoch)
    else:
        IU = nan * zeros([images.shape[0], num_classes, 2])
        for idx in range(images.shape[0]):
            img = images[idx, ...]
            lbl = labels[idx, ...]

            height = dot([256, 1], img[-2, -2:, 0])
            width = dot([256, 1], img[-1, -2:, 0])
            print('[%04d/%04d], Size: (%d, %d)' % (idx + 1, images.shape[0], height, width))

            seg_rgb = zeros((height, 3 * width, 3), dtype=uint8)
            pred = sess.run(pred_train, feed_dict={x: reshape(img, [1, 512, 512, 3])})

            seg_rgb[:, :width, :] = img[:height, :width, :]
            lbl = lbl[:height, :width]
            pred = pred[0, :height, :width]
            for k, clr in enumerate(colors):
                if k < num_classes:
                    IU[idx, k, 0] = (lbl == k) & (pred == k)    # intersection
                    IU[idx, k, 1] = (lbl == k) | (pred == k)    # union

                    IU[idx, k, 0] &= (lbl != 255)
                    IU[idx, k, 1] &= (lbl != 255)

                    seg_rgb[:, width:(2 * width), :][lbl == k] = clr
                    seg_rgb[:, (2 * width):, :][pred == k] = clr
                else:
                    seg_rgb[:, width:(2 * width), :][lbl == 255] = clr

            imsave('./results/%04d.png' % idx, seg_rgb)
        save('IU.npy', IU)
