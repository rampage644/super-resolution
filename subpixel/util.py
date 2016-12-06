'''Utilities'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import numpy as np


def generate_samples_from(image, patch_size=17, stride=13):
    '''Generate samples from HR image'''
    height, width, _ = image.shape
    h_steps, w_steps = (np.arange(0, height - stride, stride),
                        np.arange(0, width - stride, stride))

    for h_step, w_step in itertools.product(h_steps, w_steps):
        yield image[w_step:w_step + patch_size,
                    h_step:h_step + patch_size]
