#%%
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.ndimage.filters
import matplotlib.pyplot as plt

import skimage

import subpixel.model
import subpixel.util


#%%
COLORS = 3
FILTERS = [64, 32]
KERNEL_SIZES = [5, 3, 3]
FLAGS = tf.app.flags.FLAGS

#%%
path = 'data/Set5/image_SRF_2/img_001_SRF_2_bicubic.png'
ckpt = 'ckpt/'
factor = 3

def create_input(image, factor=3.0):
    shape = image.shape
    blurred = scipy.ndimage.filters.gaussian_filter(image, sigma=1)
    subsampled = scipy.misc.imresize(blurred, 1 / factor)
    return scipy.misc.imresize(subsampled, shape, interp='nearest')
#%%

image = scipy.misc.imread(path)
restored = create_input(image)

h, w, _ = image.shape

mse = ((image - restored) ** 2 / (3 * h * w)).sum()

20 * np.log(255.0 / np.sqrt(mse)) / np.log(10)

#%%
def run():
    with tf.Session() as sess:
        image = scipy.misc.imread(path)

        FLAGS.factor = factor
        FLAGS.ckpt_dir = ckpt
        FLAGS.filters = FILTERS + [FLAGS.factor ** 2 * COLORS]
        FLAGS.kernel_sizes = KERNEL_SIZES
        FLAGS.strides = [1] * len(KERNEL_SIZES)
        FLAGS.height, FLAGS.width, _ = image.shape
        FLAGS.learning_rate = 0.1

        model = subpixel.model.SuperResolution(FLAGS)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saved_model = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
        if saved_model:
            saver.restore(sess, saved_model)
        else:
            print('No saved model found, exiting.')

        sess.run(model.psnr, {
            model.input: [],
            mdeol.output: []
        })

#%%
import itertools

def show(me):
    # show first 3 images
    for i in range(1, 4):
        plt.figure(i)
        plt.imshow(me[i], interpolation='none')

r = 3  # upscaling factor
WIDTH, HEIGHT = 2, 5  # last layer spatial dimension
BATCH = 32
tshape = (BATCH, WIDTH*r, HEIGHT*r, COLORS)

clrs = [
    [123, 0, 255],
    [0, 100, 200],
    [0, 200, 100],
    [100, 100, 100],
    [200, 100, 100],
    [255, 255, 255],
    [100, 100, 200],
    [0, 56, 188],
    [10, 0, 10]
]
# we have r ^ 2 filters
for idx, clr in enumerate(clrs, start=1):
    plt.subplot(1, len(clrs), idx)
    plt.imshow(np.ones(shape=(WIDTH, HEIGHT, COLORS)) * clr)


x0 = np.concatenate([np.ones(shape=(BATCH, WIDTH, HEIGHT, COLORS)) * clr for clr in clrs], axis=3)
x0.shape  # BATCH x WIDTH x HEIGHT x COLORS * r ^ 2 -> 32x2x5x27

# here we need to reshape it so it becomes 12x15x3
# visualize to look for patterns
PS = np.zeros(shape=(BATCH, WIDTH*r, HEIGHT*r, COLORS))

plt.figure(figsize=(12, 5))

def bruteforce():
    for i, x, y, c in itertools.product(range(BATCH), range(WIDTH*r), range(HEIGHT*r), range(COLORS)):
        PS[i, x, y, c] = x0[i, x // r, y // r, COLORS * r * (y % r) + COLORS * (x % r) + c]

bruteforce()

show(PS)

# simple reshape
# obvously it doesn't work as there is no regular r x r pattern
show(x0.reshape(tshape))

# now let's reshape it with tf
# now, i'm sure it performs the same stuff as numpy
with tf.Session() as sess:
    x = tf.Variable(x0)
    x_reshaped = tf.reshape(x, shape=tshape)
    sess.run(tf.global_variables_initializer())
    show(x_reshaped.eval())

# nice!
# vsplit = split axis=0
# hstack = concatenate axis=1
x1 = np.concatenate([np.reshape(a, (-1, r * WIDTH, r, COLORS)) for a in np.split(x0, HEIGHT, axis=2)], axis=2)
x1.shape
show(x1)

# try with tf
# perfect!
with tf.Session() as sess:
    x = tf.Variable(x0)
    reshaped = tf.concat(2, [tf.reshape(a, (-1, r * WIDTH, r, COLORS)) for a in tf.split(2, HEIGHT, x)])
    sess.run(tf.global_variables_initializer())
    show(reshaped.eval())

# try to 100% replicate initial pattern (see `x0`)
arrs = np.split(x0, HEIGHT, axis=2)
arrs = np.split(x0, WIDTH, axis=1)

