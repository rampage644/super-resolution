'''Generate image'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import itertools
import operator
import tensorflow as tf
import numpy as np
import scipy.misc

import subpixel.model
import subpixel.util


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', os.path.join(os.getcwd(), 'log'), 'Directory with training data (images)')
tf.app.flags.DEFINE_string('ckpt_dir', os.path.join(os.getcwd(), 'ckpt'), 'Directory for model checkpoints')
tf.app.flags.DEFINE_integer('factor', 3, 'Upscaling factor')
tf.app.flags.DEFINE_string('input', '', 'Path to image to upscale')
tf.app.flags.DEFINE_string('output', 'output.png', 'Path to output image')

COLORS = 3
FILTERS = [64, 32]
KERNEL_SIZES = [5, 3, 3]


def main(argv=None):
    '''main entry point'''
    with tf.Session() as sess:
        image = scipy.misc.imread(FLAGS.input)

        FLAGS.filters = FILTERS + [FLAGS.factor ** 2 * COLORS]
        FLAGS.kernel_sizes = KERNEL_SIZES
        FLAGS.strides = [1] * len(KERNEL_SIZES)
        FLAGS.width, FLAGS.height, _ = image.shape
        FLAGS.learning_rate = 0.1

        model = subpixel.model.SuperResolution(FLAGS)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saved_model = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
        if saved_model:
            saver.restore(sess, saved_model)
        else:
            print('No saved model found, exiting.')
            return 1


        upscaled = sess.run(model.predicted, {
            model.input: np.expand_dims(image, axis=0)
        })

        scipy.misc.imsave(FLAGS.output, upscaled[0])


if __name__ == '__main__':
    tf.app.run()
