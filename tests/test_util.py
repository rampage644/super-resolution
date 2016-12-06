'''Utility functions tests'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import subpixel.util
import math
import numpy as np


def test_generate_samples():
    N = 100
    image = np.random.randint(255, size=(N, N, 3))
    patch_size = 17
    stride = 13
    it = subpixel.util.generate_samples_from(
        image, patch_size, stride)

    samples = list(it)

    # see math here: https://arxiv.org/pdf/1603.07285.pdf
    steps = (math.floor((N - patch_size) / stride) + 1)
    assert len(samples) == steps ** 2
    assert samples[0].shape == (patch_size, patch_size, 3)
    assert np.all(image[:patch_size, :patch_size] == samples[0])


def test_image_downscale():
    N = 100
    factor = 3
    image = np.random.randint(255, size=(N, N, 3))
    resized = subpixel.util.downscale(image, 1 / factor)

    assert resized.shape == (N // factor, N // factor, 3)
