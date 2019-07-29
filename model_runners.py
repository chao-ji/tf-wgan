import tensorflow as tf


class WGANGPTrainer(object):

  def __init__(self, prediction_model):
    self._prediction_model = prediction_model
    self._batch_size = 50
    self._noise_dim = 128
    self._lambda = 10

  def train(self, dataset, optimizer): 

    tensor_dict = dataset.get_tensor_dict()

    real_images = tensor_dict['images']


    noise = _get_random_noise()

    fake_images = self._prediction_model.generator(noise)
    images = tf.concat([real_images, fake_images], axis=0)
    scores = self._prediction_model.critic(images)

    real_scores = scores[:self._batch_size]
    fake_scores = scores[self._batch_size:]

    gp_loss = self._compute_gradient_penalty_loss(real_images, fake_images)

    critic_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores) + self._lambda * gp_loss
    generator_loss = -tf.reduce_mean(fake_scores)

    critic_update_ops, generator_update_ops = self._get_update_ops(
        optimizer, critic_loss, generator_loss)
    return critic_update_ops, generator_update_ops



  def _compute_gradient_penalty_loss(self, real_images, fake_images):
    epsilon = tf.random_uniform([self._batch_size, 1, 1, 1], 0., 1.)
    mixed_images = real_images * epsilon + fake_images * (1 - epsilon)
    scores_mixed_images = self._prediction_model.critic(mixed_images)
    grad_mixed_images = tf.gradients(scores_mixed_images, mixed_images)[0]
    gp_loss = tf.reduce_mean((tf.sqrt(tf.reduce_sum(tf.square(
        grad_mixed_images), axis=[1, 2, 3])) - 1) ** 2)

    return gp_loss


  def _get_update_ops(self, optimizer, critic_loss, generator_loss):
    generator_variables = tf.get_collection('trainable_variables', 'generator')
    critic_variables = tf.get_collection('trainable_variables', 'critic')

    critic_update_ops = optimizer.minimize(critic_loss, var_list=critic_variables)
    generator_update_ops = optimizer.minimize(generator_loss, var_list=generator_variables)
    generator_update_ops = tf.group(*([generator_update_ops] +  
        tf.get_collection(tf.GraphKeys.UPDATE_OPS)), name='update_barrier')

    return critic_update_ops, generator_update_ops

  def _get_random_noise(self, kwargs=None):
    return tf.random_normal([self._batch_size, self._noise_dim])



class WGANGPInferencer(object):
  def __init__(self, prediction_model, batch_size=50, noise_dim=128):
    self._batch_size = batch_size
    self._noise_dim = noise_dim
    self._prediction_model = prediction_model

  def infer(self, noise):
    fake_images = self._prediction_model.generator(noise) 

    return fake_images


