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
    plt.imshow(me, interpolation='none')

r = 3  # upscaling factor
WIDTH, HEIGHT = 2, 5  # last layer spatial dimension
tshape = (WIDTH*r, HEIGHT*r, COLORS)

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


x0 = np.concatenate([np.ones(shape=(WIDTH, HEIGHT, COLORS)) * clr for clr in clrs], axis=2)
x0.shape  # N-1 x N x COLORS * r ^ 2 -> 4x5x27

# here we need to reshape it so it becomes 12x15x3
# visualize to look for patterns
PS = np.zeros(shape=(WIDTH*r, HEIGHT*r, COLORS))

plt.figure(figsize=(12, 5))

def bruteforce():
    for x, y, c in itertools.product(range(WIDTH*r), range(HEIGHT*r), range(COLORS)):
        PS[x, y, c] = x0[x // r, y // r, COLORS * r * (y % r) + COLORS * (x % r) + c]

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
x1 = np.hstack([np.reshape(a, (r * WIDTH, r, COLORS)) for a in np.hsplit(x0, HEIGHT)])
show(x1)

# try with tf
# perfect!
with tf.Session() as sess:
    x = tf.Variable(x0)
    reshaped = tf.concat(1, [tf.reshape(a, (r * WIDTH, r, COLORS)) for a in tf.split(1, HEIGHT, x)])
    sess.run(tf.global_variables_initializer())
    show(reshaped.eval())
