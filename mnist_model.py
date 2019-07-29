import functools

import tensorflow as tf

slim = tf.contrib.slim


class MNISTPredictionModel(object):

  def __init__(self, 
               weights_initializer, 
               depth=64, 
               kernel_size=5, 
               leaky_relu_slope=0.2):
    self._depth = depth
    self._kernel_size = kernel_size
    self._leaky_relu_slope = leaky_relu_slope
    self._weights_initializer = weights_initializer

  def generator(self, noise, scope='generator'):
    with tf.variable_scope(scope, 'generator', [noise], reuse=tf.AUTO_REUSE):
      dense1 = slim.fully_connected(
          noise,
          num_outputs=4 * 4 * 4*self._depth,
          activation_fn=tf.nn.relu,
          weights_initializer=self._weights_initializer)
      dense1 = tf.reshape(dense1, (-1, 4, 4, 4 * self._depth))

      tconv1 = slim.conv2d_transpose(
          dense1,
          num_outputs=2 * self._depth,
          kernel_size=self._kernel_size,
          stride=2,
          padding='SAME',
          activation_fn=tf.nn.relu,
          weights_initializer=self._weights_initializer)
      tconv1 = tconv1[:, :7, :7, :]

      tconv2 = slim.conv2d_transpose(
          tconv1,
          num_outputs=self._depth,
          kernel_size=self._kernel_size,
          stride=2,
          padding='SAME',
          activation_fn=tf.nn.relu,
          weights_initializer=self._weights_initializer)

      fake_images = slim.conv2d_transpose(
          tconv2,
          num_outputs=1,
          kernel_size=self._kernel_size,
          stride=2,
          padding='SAME',
          activation_fn=tf.nn.tanh,
          normalizer_fn=None,
          normalizer_params=None,
          weights_initializer=self._weights_initializer)

    return fake_images

  def critic(self, data, scope='critic'):
    leaky_relu = functools.partial(
        tf.nn.leaky_relu, alpha=self._leaky_relu_slope)
    with tf.variable_scope(scope, 'critic', [data], reuse=tf.AUTO_REUSE):
      conv1 = slim.conv2d(data, 
                          num_outputs=self._depth, 
                          kernel_size=self._kernel_size, 
                          stride=2, 
                          padding='SAME',
                          activation_fn=leaky_relu,
                          weights_initializer=self._weights_initializer)

      conv2 = slim.conv2d(conv1, 
                          num_outputs=self._depth * 2, 
                          kernel_size=self._kernel_size, 
                          stride=2, 
                          padding='SAME',
                          activation_fn=leaky_relu,
                          weights_initializer=self._weights_initializer)

      conv3 = slim.conv2d(conv2, 
                          num_outputs=self._depth * 4, 
                          kernel_size=self._kernel_size, 
                          stride=2, 
                          padding='SAME',
                          activation_fn=leaky_relu,
                          weights_initializer=self._weights_initializer)

      reshaped = tf.reshape(conv3, (-1, 4 * 4 * 4 * self._depth))

      scores = slim.fully_connected(
          reshaped,
          num_outputs=1,
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None,
          weights_initializer=self._weights_initializer)

    return scores 
