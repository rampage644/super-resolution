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
tf.app.flags.DEFINE_integer('factor', 3, 'Upscaling factor')
tf.app.flags.DEFINE_string('input', '', 'Path to image to upscale')
tf.app.flags.DEFINE_string('output', 'output.png', 'Path to output image')

COLORS = 3
FILTERS = [64, 32]
KERNEL_SIZES = [5, 3, 3]


def main(argv=None):
    '''main entry point'''
    with tf.Session() as sess:
        FLAGS.filters = FILTERS + [FLAGS.factor ** 2 * COLORS]
        FLAGS.kernel_sizes = KERNEL_SIZES
        FLAGS.strides = [1] * len(KERNEL_SIZES)

        model = subpixel.model.SuperResolution(FLAGS)

        sess.run(tf.global_variables_initializer())

        image = scipy.misc.imread(FLAGS.input)
        upscaled = sess.run(model.predicted, {
            model.input: np.expand_dims(image, axis=0)
        })

        scipy.misc.imsave(FLAGS.output, upscaled[0])


if __name__ == '__main__':
    tf.app.run()
