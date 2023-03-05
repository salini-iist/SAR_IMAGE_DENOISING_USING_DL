import tensorflow as tf

# Nearest-neighbor upscaling layer.
def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x

regularizer = tf.contrib.layers.l2_regularizer(0.1)

def autoencoder(x, width=256, height=256, **_kwargs):
    """if config.get_nb_channels() == 1:
        x.set_shape([None, 1, height, width])
    else:
        x.set_shape([None, 3, height, width])"""
    x.set_shape([None, height, width, 1])

    skips = [x]

    n = x
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 48, 3, padding='same', name='enc_conv0'), alpha=0.1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 48, 3, padding='same', name='enc_conv1'), alpha=0.1)
    n = tf.nn.max_pool(n, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    skips.append(n)

    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 48, 3, padding='same', name='enc_conv2'), alpha=0.1)
    n = tf.nn.max_pool(n, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    skips.append(n)

    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 48, 3, padding='same', name='enc_conv3'), alpha=0.1)
    n = tf.nn.max_pool(n, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    skips.append(n)

    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 48, 3, padding='same', name='enc_conv4'), alpha=0.1)
    n = tf.nn.max_pool(n, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    skips.append(n)

    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 48, 3, padding='same', name='enc_conv5'), alpha=0.1)
    n = tf.nn.max_pool(n, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 48, 3, padding='same', name='enc_conv6'), alpha=0.1)


    # -----------------------------------------------
    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=-1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 96, 3, padding='same', name='dec_conv5'), alpha=0.1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 96, 3, padding='same', name='dec_conv5b'), alpha=0.1)

    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=-1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 96, 3, padding='same', name='dec_conv4'), alpha=0.1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 96, 3, padding='same', name='dec_conv4b'), alpha=0.1)

    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=-1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 96, 3, padding='same', name='dec_conv3'), alpha=0.1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 96, 3, padding='same', name='dec_conv3b'), alpha=0.1)

    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=-1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 96, 3, padding='same', name='dec_conv2'), alpha=0.1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 96, 3, padding='same', name='dec_conv2b'), alpha=0.1)

    n = upscale2d(n)
    n = tf.concat([n, skips.pop()], axis=-1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 64, 3, padding='same', name='dec_conv1a'), alpha=0.1)
    n = tf.nn.leaky_relu(tf.layers.conv2d(n, 32, 3, padding='same', name='dec_conv1b'), alpha=0.1)

    n = tf.layers.conv2d(n, 1, 3, padding='same', name='dec_conv1')

    return x - n
