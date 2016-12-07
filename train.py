'''Train the model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import itertools
import operator
import tensorflow as tf
import numpy as np

import subpixel.model
import subpixel.util


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', os.getcwd(), 'Directory with training data (images)')
tf.app.flags.DEFINE_string('log_dir', os.path.join(os.getcwd(), 'log'), 'Directory with training data (images)')
tf.app.flags.DEFINE_integer('factor', 3, 'Upscaling factor')
tf.app.flags.DEFINE_integer('patch_size', 17, 'Patch size to crop images into')
tf.app.flags.DEFINE_integer('height', 17, 'Patch height')
tf.app.flags.DEFINE_integer('width', 17, 'Patch widht')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_integer('epoch', 10, 'Epoch count')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')

COLORS = 3
FILTERS = [64, 32]
KERNEL_SIZES = [5, 3, 3]


# Taken from python `itertools` recipes
def grouper(iterable, number, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * number
    return zip(*args)


def main(argv=None):
    '''main entry point'''
    with tf.Session() as sess:
        FLAGS.filters = FILTERS + [FLAGS.factor ** 2 * COLORS]
        FLAGS.kernel_sizes = KERNEL_SIZES
        FLAGS.strides = [1] * len(KERNEL_SIZES)

        model = subpixel.model.SuperResolution(FLAGS)
        patch_stride = FLAGS.patch_size - sum(map(lambda x: x % 2, KERNEL_SIZES))

        sess.run(tf.global_variables_initializer())
        writer = tf.train.SummaryWriter(FLAGS.log_dir, graph=tf.get_default_graph())

        for epoch in range(FLAGS.epoch):
            for data in grouper(
                    subpixel.util.generate_train_data_from_dir(
                        FLAGS.train_dir, FLAGS.factor, FLAGS.patch_size, patch_stride),
                    FLAGS.batch_size
            ):
                xdata, ydata = (
                    list(map(operator.itemgetter(0), data)),
                    list(map(operator.itemgetter(1), data))
                )
                loss, psnr, summary, _ = sess.run([
                        model.loss,
                        model.psnr,
                        model.summary,
                        model.train_op], {
                    model.input: np.array(xdata, ndmin=4),
                    model.output: np.array(ydata, ndmin=4)
                })

                writer.add_summary(summary, model.step.eval(sess))

            print('\rEpoch {}: loss: {:.2f} psnr {:.2f}'.format(epoch+1, loss, psnr), end='')
    print()


if __name__ == '__main__':
    tf.app.run()
