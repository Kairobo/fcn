from util import *
import tensorflow as tf
import time, os
from numpy import *

training = True
batch_size = 8
lr_basic = 1e-4
num_epoch = 10

x = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='x')
y = tf.placeholder(tf.int64, shape=[None, 512, 512], name='y')

logits, loss = build_model(x, y, training=training)

vars_trainable = tf.trainable_variables()
for var in vars_trainable:
    print(var.name, var.get_shape())
    tf.summary.histogram(var.name, var)

tf.summary.scalar('loss', loss)

result = tf.concat([y, tf.argmax(logits, axis=-1)], axis=2)
result = tf.cast(12 * tf.reshape(result, [-1, 512, 1024, 1]), tf.uint8)

tf.summary.image('result', result, max_outputs=8)

sum_all = tf.summary.merge_all()

####
print('\nLoading data ...')
t0 = time.time()
if training:
    images = load_images('./data/train/images/*.png')
    labels = load_images('./data/train/labels/*.png')
else:
    images = load_images('./data/val/images/*.png')
    labels = load_images('./data/val/labels/*.png')
print('Finished loading in %.2f seconds.' % (time.time() - t0))

order = arange(images.shape[0])

lr = tf.placeholder(tf.float32, shape=[], name='lr')
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=vars_trainable)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    if training:
        writer = tf.summary.FileWriter("./logs", sess.graph)

        total_count = 0
        t0 = time.time()
        for epoch in range(num_epoch):
            random.shuffle(order)
            for k in range(images.shape[0] // batch_size):
                idx = order[(k * batch_size):min((k + 1) * batch_size, 1 + images.shape[0])]
                img = images[idx, ...]
                lbl = labels[idx, ...]

                if random.rand() > 0.5:
                    img = img[:, :, ::-1, :]
                    lbl = lbl[:, :, ::-1]

                l_, _ = sess.run([loss, train_step],
                                feed_dict={x: img, y: lbl, lr: lr_basic / 5**epoch})

                total_count += 1
                writer.add_summary(sess.run(sum_all, feed_dict={x: img, y: lbl}), total_count)

                m, s = divmod(time.time() - t0, 60)
                h, m = divmod(m, 60)
                print('Epoch: [%4d/%4d] [%4d/%4d], Time: [%02d:%02d:%02d], loss: %.4f'
                        % (epoch, num_epoch, k, len(images) // batch_size, h, m, s, l_))

            print('Saving checkpoint ...')
            saver.save(sess, './checkpoint/FCN.ckpt', global_step=total_count)
    else:
        ckpt = tf.train.get_checkpoint_state('./checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join('./checkpoint', ckpt_name))
            print(' [*] Success to read {}'.format(ckpt_name))
        else:
            raise ValueError(' [*] Failed to find a checkpoint')
