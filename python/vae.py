#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from collections import defaultdict

from utils import *  # noqa
from itertools import chain


class VAE(object):
    def __init__(self, x, dataset, n_input, n_z, network_architecture, learning_rate,
                 encoder_distribution='multinomial', decoder_distribution='multinomial',
                 nonlinearity=tf.nn.softplus, n_ary=None, n_samples=1, train_bias=None):
        self.dataset = dataset
        self.n_input = n_input
        self.n_z = n_z
        self.network_architecture = network_architecture
        self.learning_rate_value = learning_rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.encoder_distribution = encoder_distribution
        self.decoder_distribution = decoder_distribution
        self.nonlinearity = nonlinearity
        self.n_ary = n_ary
        self.n_samples_value = n_samples
        self.n_samples = tf.placeholder(tf.int32, shape=[])
        self.is_train = tf.placeholder(tf.int32, shape=[])
        self.train_bias = train_bias

        with tf.name_scope('input'):
            self.x_binarized = x
            self.batch_size = tf.shape(self.x_binarized)[0]
            self.x_binarized = tf.tile(self.x_binarized, [1, self.n_samples, 1])
        with tf.name_scope('prior_probs'):
            self.prior_probs = (1. / self.n_ary) * \
                tf.ones([self.batch_size, self.n_samples, self.n_ary * self.n_z])

        self.__create_network()

        self.__create_loss_optimizer()

        self.init_weights = tf.variables_initializer(self.flat_weights.values())
        self.saver = tf.train.Saver(self.flat_weights, max_to_keep=None)

    def __str__(self):
        return 'ReparamTrickVAE'

    def _initialize_weights(self):
        weights = defaultdict(dict)
        for nn_part, layers in self.network_architecture.items():
            for name, (n_in, n_out) in layers.items():
                weights[nn_part][name] = init_weights(n_in, n_out, name=nn_part + '_' + name)
        self.encoder_weights = list(chain.from_iterable(weights['encoder'].values()))
        self.decoder_weights = list(chain.from_iterable(weights['decoder'].values()))
        self.flat_weights = flatten(weights)
        return weights

    def __create_network(self):
        self.weights = self._initialize_weights()
        self._create_encoder_part()
        self.x_reconst = self._create_decoder_part(self.z)

    def _create_encoder_part(self):
        weights = self.weights
        x = self.x_binarized
        x, sh = unroll_tensor(x)

        with tf.name_scope('encoder'):
            self.encoder_layer1 = build_layer(x, *weights['encoder']['h1'], nonlinearity=self.nonlinearity,
                                              name='encoder_h1')
            self.encoder_layer2 = build_layer(self.encoder_layer1, *weights['encoder']['h2'],
                                              nonlinearity=self.nonlinearity, name='encoder_h2')
        with tf.name_scope('stochastic_layer'):
            self.z_mean = build_layer(self.encoder_layer2, *
                                      weights['encoder']['out_mean'], name='encoder_out_mean')
            self.z_mean = roll_tensor(self.z_mean, sh)

            if self.encoder_distribution == 'gaussian':
                self.z_log_sigma_sq = build_layer(self.encoder_layer2, *weights['encoder']['out_log_sigma_sq'],
                                                  name='encoder_out_log_sigma_sq')

                self.z_log_sigma_sq = roll_tensor(self.z_log_sigma_sq, sh)
                with tf.name_scope('stochastic_node'):
                    epsilon = tf.random_normal(tf.shape(self.z_mean))
                    self.z = self.z_mean + tf.exp(0.5 * self.z_log_sigma_sq) * epsilon
            elif self.encoder_distribution == 'multinomial':
                with tf.name_scope('reshape'):
                    self.z_mean = tf.reshape(self.z_mean, [-1, self.n_ary])
                with tf.name_scope('stochastic_node'):
                    self.z = tf.one_hot(tf.squeeze(tf.multinomial(
                        self.z_mean, 1), 1), self.n_ary, 1.0, 0.0)
                    self.z = tf.reshape(
                        self.z, [self.batch_size, self.n_samples, self.n_ary * self.n_z])
                with tf.name_scope('z_probs'):
                    self.z_probs = tf.reshape(tf.nn.softmax(self.z_mean),
                                              [self.batch_size, self.n_samples, self.n_ary * self.n_z])
                with tf.name_scope('reshape'):
                    self.z_mean = tf.reshape(
                        self.z_mean, [self.batch_size, self.n_samples, self.n_ary * self.n_z])

    def _create_decoder_part(self, z):
        weights = self.weights
        with tf.name_scope('decoder'):
            z, sh = unroll_tensor(z)
            decoder_layer1 = build_layer(z, *weights['decoder']['h1'], nonlinearity=self.nonlinearity,
                                         name='decoder_h1')
            decoder_layer2 = build_layer(decoder_layer1, *weights['decoder']['h2'], nonlinearity=self.nonlinearity,
                                         name='decoder_h2')
            x_reconst = build_layer(
                decoder_layer2, *weights['decoder']['out_mean'], name='decoder_out_mean')
            x_reconst = roll_tensor(x_reconst, sh)
        return x_reconst

    def __create_loss_optimizer(self):
        self._create_loss()
        self._create_optimizer()

    def _create_loss(self):
        with tf.name_scope('loss'):
            if self.decoder_distribution == 'multinomial':
                self.decoder_log_density = compute_log_density(x=self.x_binarized, logits=self.x_reconst,
                                                               distribution=self.decoder_distribution,
                                                               name='decoder_log_density')
            elif self.decoder_distribution == 'gaussian':
                self.decoder_log_density = compute_log_density(x=self.x_binarized, mu=tf.nn.softmax(self.x_reconst),
                                                               sigma=1., distribution=self.decoder_distribution,
                                                               name='decoder_log_density')
            if self.encoder_distribution == 'multinomial':
                with tf.name_scope('reshape'):
                    z = tf.reshape(self.z, [self.batch_size, self.n_samples, self.n_z, self.n_ary])
                    z_shape = tf.shape(z)
                    z_mean, z_probs, prior_probs = tuple(map(lambda x: tf.reshape(x, z_shape),
                                                             [self.z_mean, self.z_probs, self.prior_probs]))
                self.encoder_log_density = compute_log_density(x=z, logits=z_mean, probs=z_probs,
                                                               prior_probs=prior_probs,
                                                               distribution=self.encoder_distribution,
                                                               name='encoder_log_density')
                self.kl_divergency = compute_kl_divergency(x=z, logits=z_mean, probs=z_probs,
                                                           prior_probs=prior_probs,
                                                           distribution=self.encoder_distribution,
                                                           name='kl_divergency')
            elif self.encoder_distribution == 'gaussian':
                self.encoder_log_density = compute_log_density(x=self.z, mu=self.z_mean,
                                                               sigma=tf.exp(
                                                                   0.5 * self.z_log_sigma_sq),
                                                               distribution=self.encoder_distribution,
                                                               name='encoder_log_density')
                self.kl_divergency = compute_kl_divergency(x=self.z, mu=self.z_mean,
                                                           sigma=tf.exp(0.5 * self.z_log_sigma_sq),
                                                           distribution=self.encoder_distribution,
                                                           name='kl_divergency')
            self.multisample_elbo = - \
                compute_multisample_elbo(self.decoder_log_density, self.kl_divergency)

            self.cost_for_decoder_weights = tf.reduce_mean(self.multisample_elbo)
            self.cost_for_encoder_weights = tf.reduce_mean(self.multisample_elbo)

            self.cost_for_display = tf.reduce_mean(self.multisample_elbo)

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            with tf.name_scope('decoder_optimizer'):
                self.decoder_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, epsilon=1e-4)
                self.decoder_minimizer = self.decoder_optimizer.minimize(self.cost_for_decoder_weights,
                                                                         var_list=self.decoder_weights)
            with tf.name_scope('encoder_optimizer'):
                self.encoder_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, epsilon=1e-4)
                self.encoder_minimizer = self.encoder_optimizer.minimize(self.cost_for_encoder_weights,
                                                                         var_list=self.encoder_weights)
            with tf.name_scope('gradients'):
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, epsilon=1e-4)
                self.decoder_gradients = self.optimizer.compute_gradients(self.cost_for_decoder_weights,
                                                                          var_list=self.decoder_weights)
                self.encoder_gradients = self.optimizer.compute_gradients(self.cost_for_encoder_weights,
                                                                          var_list=self.encoder_weights)

    def initialize_weights(self, sess):
        sess.run(self.init_weights)

    def partial_fit(self, learning_rate_decay=1.0,
                    n_samples=None, is_train=True):
        learning_rate = learning_rate_decay * self.learning_rate_value
        n_samples = self.n_samples_value if n_samples is None else n_samples
        dict_of_tensors = {
            'decoder_minimizer': self.decoder_minimizer,
            'encoder_minimizer': self.encoder_minimizer,
            'cost_for_display': self.cost_for_display
        }
        feed_dict = {self.learning_rate: learning_rate,
                     self.n_samples: n_samples,
                     self.is_train: 1 if is_train else 0}
        return dict_of_tensors, feed_dict

    def name(self):
        return str(self)

    def dataset_name(self):
        return self.dataset

    def parameters(self):
        return '{}-{}n_ary-m{}-nz{}-lr{:.5f}'.format(self.encoder_distribution, self.n_ary,
                                                     self.n_samples_value, self.n_z, self.learning_rate_value)

    def get_decoder_gradients(self, X, learning_rate=None, n_samples=None):
        learning_rate = self.learning_rate_value if learning_rate is None else learning_rate
        n_samples = self.n_samples_value if n_samples is None else n_samples
        gradients = self.sess.run(self.decoder_gradients, feed_dict={self.x: X,
                                                                     self.learning_rate: learning_rate,
                                                                     self.n_samples: n_samples})
        flat_gradients = np.array([])
        for grad in gradients:
            flat_gradients = np.append(flat_gradients, grad[0].flatten())
        return flat_gradients

    def get_encoder_gradients(self, X, learning_rate=None, n_samples=None):
        learning_rate = self.learning_rate_value if learning_rate is None else learning_rate
        n_samples = self.n_samples_value if n_samples is None else n_samples
        gradients = self.sess.run(self.encoder_gradients, feed_dict={self.x: X,
                                                                     self.learning_rate: learning_rate,
                                                                     self.n_samples: n_samples})
        flat_gradients = np.array([])
        for grad in gradients:
            flat_gradients = np.append(flat_gradients, grad[0].flatten())
        return flat_gradients

    def get_decoder_log_density(self, X):
        return self.sess.run(tf.reduce_mean(self.decoder_log_density), feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z=None):
        if z is None:
            z = np.random.normal(size=self.n_z)
        return self.sess.run(self.x_reconst, feed_dict={self.z: z})

    def reconstruct(self, X):
        return self.sess.run(self.x_reconst, feed_dict={self.x: X})

    def loss(self, n_samples=None):
        n_samples = self.n_samples_value if n_samples is None else n_samples
        dict_of_tensors = {
            'cost_for_display': self.cost_for_display
        }
        feed_dict = {self.n_samples: n_samples}
        return dict_of_tensors, feed_dict

    def save_weights(self, sess, save_path):
        self.saver.save(sess, save_path)

    def restore_weights(self, sess, restore_path):
        self.saver.restore(sess, restore_path)

    def close(self):
        self.sess.close()


class LogDerTrickVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._create_network()

        self._create_loss_optimizer()

    def __str__(self):
        return 'LogDerTrickVAE'

    def _create_network(self):
        self.z = tf.stop_gradient(self.z)
        self.x_reconst = self._create_decoder_part(self.z)

    def _create_loss_optimizer(self):
        self._create_loss()
        if self.n_samples_value == 1:
            self.log_der_trick_cost = - self.encoder_log_density * self.decoder_log_density
            self.cost_for_encoder_weights = tf.reduce_mean(
                self.kl_divergency + self.log_der_trick_cost)
        elif self.n_samples_value > 1:
            encoder_log_density = tf.reduce_sum(self.encoder_log_density, 1)
            self.log_der_trick_cost = encoder_log_density * tf.stop_gradient(self.multisample_elbo)
            self.log_der_trick_cost += self.multisample_elbo
            self.cost_for_encoder_weights = tf.reduce_mean(self.log_der_trick_cost)
        self._create_optimizer()


class VIMCOVAE(VAE):
    def __init__(self, *args, n_samples=2, **kwargs):
        super().__init__(*args, n_samples=n_samples, **kwargs)

        self._create_network()

        self._create_loss_optimizer()

    def __str__(self):
        return 'VIMCOVAE'

    def _create_network(self):
        self.z = tf.stop_gradient(self.z)
        self.x_reconst = self._create_decoder_part(self.z)

    def _create_loss_optimizer(self):
        self._create_loss()
        self.multisample_elbo = - \
            compute_multisample_elbo(self.decoder_log_density, self.kl_divergency)
        self.multisample_elbo_vimco = -compute_multisample_elbo(self.decoder_log_density,
                                                                self.kl_divergency, is_vimco=True)
        self.multisample_elbo_vimco = tf.stop_gradient(tf.transpose(self.multisample_elbo_vimco))
        self.vimco_cost = tf.reduce_sum(self.encoder_log_density * self.multisample_elbo_vimco, 1)
        self.vimco_cost += self.multisample_elbo
        self.cost_for_encoder_weights = tf.reduce_mean(self.vimco_cost)
        self._create_optimizer()


class NVILVAE(VAE):
    def __init__(self, *args, baseline_learning_rate=1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_learning_rate = baseline_learning_rate

        self._create_network()

        self._create_loss_optimizer()

    def __str__(self):
        return 'NVILVAE'

    def _create_network(self):
        self.z = tf.stop_gradient(self.z)
        self.x_reconst = self._create_decoder_part(self.z)

        n_hidden_encoder_2 = self.network_architecture['encoder']['h2'][1]
        self.baseline_weights = init_weights(n_hidden_encoder_2, 1, name='baseline')
        self.baseline = build_layer(tf.stop_gradient(self.encoder_layer2),
                                    *self.baseline_weights)
        self.baseline_weights = list(self.baseline_weights)
        self.baseline_saver = tf.train.Saver(self.baseline_weights, max_to_keep=None)

    def _create_loss_optimizer(self):
        self._create_loss()

        if self.n_samples_value == 1:
            self.nvil_cost = - self.encoder_log_density * (self.decoder_log_density - self.baseline)
            self.cost_for_encoder_weights = tf.reduce_mean(self.kl_divergency + self.nvil_cost)
            self.cost_for_baseline = tf.reduce_mean((self.decoder_log_density - self.baseline)**2)
        elif self.n_samples_value > 1:
            encoder_log_density = tf.reduce_sum(self.encoder_log_density, 1)
            self.nvil_cost = encoder_log_density * \
                tf.stop_gradient((self.multisample_elbo - self.baseline))
            self.nvil_cost += self.multisample_elbo
            self.cost_for_encoder_weights = tf.reduce_mean(self.nvil_cost)
            self.cost_for_baseline = tf.reduce_mean((self.multisample_elbo - self.baseline)**2)
        self._create_optimizer()

        self.baseline_optimizer = tf.train.AdamOptimizer(learning_rate=self.baseline_learning_rate)
        self.baseline_minimizer = self.decoder_optimizer.minimize(self.cost_for_baseline,
                                                                  var_list=self.baseline_weights)

    def partial_fit(self, learning_rate_decay=1.0, n_samples=None):
        dict_of_tensors, feed_dict = super().partial_fit(learning_rate_decay, n_samples)
        dict_of_tensors['baseline_minimizer'] = self.baseline_minimizer
        return dict_of_tensors, feed_dict

    def save_weights(self, sess, save_path):
        super().save_weights(sess, save_path)
        self.baseline_saver.save(sess, save_path + '_baseline')

    def restore_weights(self, sess, restore_path):
        super().restore_weights(sess, restore_path)
        self.baseline_saver.restore(sess, restore_path + '_baseline')


class MuPropVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._create_network()

        self._create_loss_optimizer()

    def __str__(self):
        return 'MuPropVAE'

    def _create_network(self):
        self.z = tf.stop_gradient(self.z)
        self.x_reconst = self._create_decoder_part(self.z)
        if self.encoder_distribution == 'multinomial':
            self.z_mean = tf.reshape(
                self.z_mean, [self.batch_size, self.n_samples, self.n_ary * self.n_z])

        self.x_reconst_mean = self._create_decoder_part(self.z_mean)

    def _create_loss_optimizer(self):
        self._create_loss()

        if self.decoder_distribution == 'multinomial':
            self.decoder_log_density_mean = compute_log_density(x=self.x_binarized, logits=self.x_reconst_mean,
                                                                distribution=self.decoder_distribution)
        elif self.decoder_distribution == 'gaussian':
            self.decoder_log_density_mean = compute_log_density(x=self.x_binarized,
                                                                mu=tf.nn.softmax(self.x_reconst_mean), sigma=1.,
                                                                distribution=self.decoder_distribution)

        if self.n_samples_value == 1:
            jacobian = tf.gradients(self.decoder_log_density_mean, self.z_mean)[0]
            self.decoder_log_density_mean = tf.stop_gradient(self.decoder_log_density_mean)
        elif self.n_samples_value > 1:
            self.multisample_elbo_mean = - \
                compute_multisample_elbo(self.decoder_log_density_mean, self.kl_divergency)
            jacobian = tf.gradients(self.multisample_elbo_mean, self.z_mean)[0]
            self.multisample_elbo_mean = tf.stop_gradient(self.multisample_elbo_mean)
        self.jacobian = tf.stop_gradient(jacobian)

        self.linear_part = tf.stop_gradient(tf.reduce_sum(
            self.jacobian * (self.z - self.z_mean), axis=-1))
        self.deterministic_term = tf.reduce_sum(self.jacobian * self.z_mean, axis=-1)

        if self.n_samples_value == 1:
            self.decoder_log_density_adjusted = self.decoder_log_density - self.decoder_log_density_mean
            self.decoder_log_density_adjusted -= self.linear_part
            self.muprop_cost = - self.encoder_log_density * self.decoder_log_density_adjusted
            self.muprop_cost -= self.deterministic_term
            self.cost_for_encoder_weights = tf.reduce_mean(self.kl_divergency + self.muprop_cost)
        elif self.n_samples_value > 1:
            self.multisample_elbo_adjusted = self.multisample_elbo - self.multisample_elbo_mean
            self.multisample_elbo_adjusted -= tf.reduce_sum(self.linear_part, -1)
            self.multisample_elbo_adjusted = tf.stop_gradient(self.multisample_elbo_adjusted)
            self.muprop_cost = tf.reduce_sum(
                self.encoder_log_density, 1) * self.multisample_elbo_adjusted
            self.muprop_cost += tf.reduce_sum(self.deterministic_term, -1)
            self.muprop_cost += self.multisample_elbo
            self.cost_for_encoder_weights = tf.reduce_mean(self.muprop_cost)
        self._create_optimizer()


class GumbelSoftmaxTrickVAE(VAE):
    def __init__(self, *args, temperature=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

        if self.encoder_distribution != 'multinomial':
            error_message = ' does not support the {} encoder distribution'.format(
                self.encoder_distribution)
            raise ValueError(str(self) + error_message)

        self._create_network()

        self._create_loss_optimizer()

    def __str__(self):
        return 'GumbelSoftmaxTrickVAE'

    def _create_network(self):
        self.z_mean = tf.reshape(self.z_mean, [-1, self.n_ary])
        self.z_probs = tf.reshape(tf.nn.softmax(self.z_mean),
                                  [self.batch_size, self.n_samples, self.n_ary * self.n_z])

        def true_fn(z):
            return gumbel_softmax(z, self.temperature)

        def false_fn(z):
            return tf.one_hot(tf.squeeze(tf.multinomial(z, 1), 1), self.n_ary, 1.0, 0.0)

        self.z = tf.cond(tf.equal(self.is_train, tf.constant(1)), true_fn, false_fn)

        self.z = tf.reshape(self.z,
                            [self.batch_size, self.n_samples, self.n_ary * self.n_z])

        self.x_reconst = self._create_decoder_part(self.z)

    def _create_loss_optimizer(self):
        self._create_loss()
        self._create_optimizer()
