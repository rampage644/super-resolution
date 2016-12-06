'''Utility functions tests'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import unittest.mock
import operator
import numpy as np
import subpixel.util


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


def test_generate_train_data():
    N = 300
    factor = 3
    patch_size = 17
    stride = 13
    image = np.random.randint(255, size=(N, N, 3))
    o_mock = unittest.mock.MagicMock(
        spec='scipy.misc.imread', return_value=image
    )
    with unittest.mock.patch('scipy.misc.imread', o_mock):
        it = subpixel.util.generate_train_data_from(
            'somefile', factor, patch_size, stride)
        data = list(it)
        x_data, y_data = (
            list(map(operator.itemgetter(0), data)),
            list(map(operator.itemgetter(1), data))
        )

        assert len(x_data) == len(y_data)
        assert x_data[0].shape == (patch_size, patch_size, 3)
        assert y_data[0].shape == (patch_size * factor, patch_size * factor, 3)


def test_generate_data_from_directory():
    N = 51
    factor = 3
    patch_size = 17
    stride = 13
    image = np.random.randint(255, size=(N, N, 3))
    o_mock = unittest.mock.MagicMock(
        spec='scipy.misc.imread', return_value=image
    )
    listdir_mock = unittest.mock.MagicMock(
        return_value=['file1.png', 'file2.png', 'file3.txt']
    )
    with unittest.mock.patch('scipy.misc.imread', o_mock), \
         unittest.mock.patch('os.listdir', listdir_mock):
          it = subpixel.util.generate_train_data_from_dir(
              'somedir', factor, patch_size, stride)

          data = list(it)

          assert len(data) == 2
          assert len(data[0]) == 2
