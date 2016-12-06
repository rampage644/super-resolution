'''Train the model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import itertools
import tensorflow as tf

import subpixel.model
import subpixel.util


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', os.getcwd(), 'Directory with training data (images)')
tf.app.flags.DEFINE_integer('factor', 3, 'Upscaling factor')
tf.app.flags.DEFINE_integer('patch_size', 17, 'Patch size to crop images into')
tf.app.flags.DEFINE_integer('height', 17, 'Patch height')
tf.app.flags.DEFINE_integer('width', 17, 'Patch widht')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_integer('epoch', 10, 'Epoch count')

COLORS = 3
FILTERS = [64, 32]
KERNEL_SIZES = [5, 3, 3]


# Taken from python `itertools` recipes
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def main(argv=None):
    '''main entry point'''
    with tf.Session() as sess:
        FLAGS.filters = FILTERS + [FLAGS.factor ** 2 * COLORS]
        FLAGS.kernel_sizes = KERNEL_SIZES
        FLAGS.strides = [1] * len(KERNEL_SIZES)

        model = subpixel.model.SuperResolution(FLAGS)
        patch_stride = sum(map(lambda x: x % 2, KERNEL_SIZES))

        for e in range(FLAGS.epoch):
            for data in grouper(
                subpixel.util.generate_train_data_from_dir(FLAGS.train_dir, FLAGS.factor, FLAGS.patch_size, patch_stride),
                FLAGS.batch_size
            ):
                import ipdb; ipdb.set_trace()
                loss, _ = sess.run([model.loss, model.train_op], {
                    model.input: x,
                    model.output: y
                })

                print('\rloss: {}'.format(loss), end='')


if __name__ == '__main__':
    tf.app.run()

