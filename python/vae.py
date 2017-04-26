import numpy as np
import tensorflow as tf

from utils import *
from itertools import chain

class VAE(object):
    def __init__(self, n_input, n_z, network_architecture, learning_rate, 
                 encoder_distribution='multinomial', decoder_distribution='multinomial', n_ary=None):
        self.n_input = n_input
        self.n_z = n_z
        self.network_architecture = network_architecture
        self.learning_rate_value = learning_rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.encoder_distribution = encoder_distribution
        self.decoder_distribution = decoder_distribution
        self.n_ary = n_ary
        
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.x_binarized = tf.cast(tf.random_uniform(tf.shape(self.x)) <= self.x, tf.float32)
        
        self.prior_probs = 0.5*tf.ones([tf.shape(self.x)[0], self.n_ary*self.n_z])
        
        self.__create_network()
        
        self.__create_loss_optimizer()
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(self.flat_weights, max_to_keep=None)
        
        self.sess = tf.Session()
        self.sess.run(self.init)
        
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
        
        self.encoder_layer1 = build_layer(x, *weights['encoder']['h1'], nonlinearity=tf.nn.softplus)
        self.encoder_layer2 = build_layer(self.encoder_layer1, *weights['encoder']['h2'], nonlinearity=tf.nn.softplus)
        self.z_mean = build_layer(self.encoder_layer2, *weights['encoder']['out_mean'])
        
        if self.encoder_distribution == 'gaussian':
            self.z_log_sigma_sq = build_layer(self.encoder_layer2, *weights['encoder']['out_log_sigma_sq'])
            epsilon = tf.random_normal((tf.shape(self.x)[0], self.n_z), 0, 1, dtype=tf.float32)
            self.z = self.z_mean + tf.exp(0.5 * self.z_log_sigma_sq) * epsilon
        elif self.encoder_distribution == 'multinomial':
            self.z_mean = tf.reshape(self.z_mean, [tf.shape(x)[0]*self.n_z, self.n_ary])
            self.z = tf.one_hot(tf.squeeze(tf.multinomial(self.z_mean, 1), 1), self.n_ary, 1.0, 0.0)
            self.z = tf.reshape(self.z, [tf.shape(x)[0], self.n_ary*self.n_z])
            
            self.z_probs = tf.reshape(tf.nn.softmax(self.z_mean), [tf.shape(x)[0], self.n_ary*self.n_z])
    
    def _create_decoder_part(self, z):
        weights = self.weights
        decoder_layer1 = build_layer(z, *weights['decoder']['h1'], nonlinearity=tf.nn.softplus)
        decoder_layer2 = build_layer(decoder_layer1, *weights['decoder']['h2'], nonlinearity=tf.nn.softplus)
        x_reconst = build_layer(decoder_layer2, *weights['decoder']['out_mean'], nonlinearity=tf.nn.sigmoid)
        return x_reconst
    
    def __create_loss_optimizer(self):
        self._create_loss()
        self._create_optimizer()
    
    def _create_loss(self):
        params = {}
        if self.decoder_distribution == 'multinomial':
            params['x'], params['probs'] = self.x, self.x_reconst
        elif self.decoder_distribution == 'gaussian':
            params['x'], params['mu'], params['sigma'] = self.x, self.x_reconst, 1.
        self.decoder_log_density = log_density(params, self.decoder_distribution)
        
        params = {}
        if self.encoder_distribution == 'multinomial':
            params['x'], params['probs'], params['prior_probs'] = self.z, self.z_probs, self.prior_probs
        elif self.encoder_distribution == 'gaussian':
            params['x'], params['mu'], params['sigma'] = self.z, self.z_mean, tf.exp(0.5 * self.z_log_sigma_sq)
        self.encoder_log_density = log_density(params, self.encoder_distribution)
            
        self.kl_divergency = kl_divergency(params, distribution=self.encoder_distribution)
        
        self.cost_for_decoder_weights = tf.reduce_mean(self.kl_divergency - self.decoder_log_density)
        self.cost_for_encoder_weights = tf.reduce_mean(self.kl_divergency - self.decoder_log_density)
        
        self.cost_for_display = tf.reduce_mean(self.kl_divergency - self.decoder_log_density)
        
    def _create_optimizer(self):
        self.decoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.decoder_minimizer = self.decoder_optimizer.minimize(self.cost_for_decoder_weights, 
                                                                 var_list=self.decoder_weights)
        
        self.encoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.encoder_minimizer = self.encoder_optimizer.minimize(self.cost_for_encoder_weights, 
                                                                 var_list=self.encoder_weights)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.decoder_gradients = self.optimizer.compute_gradients(self.cost_for_decoder_weights, 
                                                                  var_list=self.decoder_weights)
        self.encoder_gradients = self.optimizer.compute_gradients(self.cost_for_encoder_weights, 
                                                                  var_list=self.encoder_weights)
        
    def partial_fit(self, X, epoch=None, learning_rate_decay=1.0):
        learning_rate = learning_rate_decay * self.learning_rate_value
        self.sess.run(self.decoder_minimizer, feed_dict={self.x: X, self.learning_rate: learning_rate})
        self.sess.run(self.encoder_minimizer, feed_dict={self.x: X, self.learning_rate: learning_rate})
    
    def get_decoder_gradients(self, X):
        gradients = self.sess.run(self.decoder_gradients, feed_dict={self.x: X})
        flat_gradients = np.array([])
        for grad in gradients:
            flat_gradients = np.append(flat_gradients, grad[0].flatten())
        return flat_gradients
    
    def get_encoder_gradients(self, X):
        gradients = self.sess.run(self.encoder_gradients, feed_dict={self.x: X})
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
    
    def loss(self, X):
        return self.sess.run(self.cost_for_display, feed_dict={self.x: X})
    
    def save_weights(self, save_path):
        self.saver.save(self.sess, save_path)
        
    def restore_weights(self, restore_path):
        self.saver.restore(self.sess, restore_path)
    
    def close(self):
        self.sess.close()

class LogDerTrickVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._create_network()
        
        self._create_loss_optimizer()
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
    def __str__(self):
        return 'LogDerTrickVAE'
        
    def _create_network(self):
        self.z = tf.stop_gradient(self.z)
        self.x_reconst = self._create_decoder_part(self.z)
    
    def _create_loss_optimizer(self):
        self._create_loss()
        self.log_der_trick_cost = - self.encoder_log_density * self.decoder_log_density
        self.cost_for_encoder_weights = tf.reduce_mean(self.kl_divergency + self.log_der_trick_cost)
        self._create_optimizer()

class VIMCOVAE(VAE):
    def __init__(self, *args, n_vimco_samples=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_vimco_samples = n_vimco_samples
        
        self._create_network()
        
        self._create_loss_optimizer()
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
    def __str__(self):
        return 'VIMCOVAE'
        
    def _create_network(self):
        self.z = tf.stop_gradient(self.z)
        self.x_reconst = self._create_decoder_part(self.z)
        
        self.z_mean_vimco = self.z_mean
        self.z_mean_vimco = tf.tile(tf.reshape(self.z_mean_vimco, [-1]), [self.n_vimco_samples])
        
        if self.encoder_distribution == 'gaussian':
            self.z_mean_vimco = tf.reshape(self.z_mean_vimco, [self.n_vimco_samples, -1, self.n_z])
            self.z_log_sigma_sq_vimco = self.z_log_sigma_sq
            self.z_log_sigma_sq_vimco = tf.tile(tf.reshape(self.z_log_sigma_sq_vimco, [-1]), [self.n_vimco_samples])
            self.z_log_sigma_sq_vimco = tf.reshape(self.z_log_sigma_sq_vimco, [self.n_vimco_samples, -1, self.n_z])
            self.z_vimco = tf.random_normal((self.n_vimco_samples, tf.shape(self.x)[0], self.n_z), 
                                            self.z_mean, tf.sqrt(tf.exp(self.z_log_sigma_sq)), dtype=tf.float32)
            self.z_vimco = tf.reshape(self.z_vimco, [self.n_vimco_samples*tf.shape(self.x)[0], self.n_z])
        elif self.encoder_distribution == 'multinomial':
            self.z_mean_vimco = tf.reshape(self.z_mean_vimco, [-1, self.n_ary])
            self.z_vimco = tf.one_hot(tf.squeeze(tf.multinomial(self.z_mean_vimco, 1), 1), self.n_ary, 1.0, 0.0)
            self.z_vimco = tf.reshape(self.z_vimco, [self.n_vimco_samples*tf.shape(self.x)[0], self.n_ary*self.n_z])

        self.x_reconst_vimco = self._create_decoder_part(self.z_vimco)
        self.x_reconst_vimco = tf.reshape(self.x_reconst_vimco, 
                                          [self.n_vimco_samples, tf.shape(self.x)[0], self.n_input])
    
    def _create_loss_optimizer(self):
        self._create_loss()
        self.x_repeated = tf.tile(tf.reshape(self.x, [-1]), [self.n_vimco_samples])
        self.x_repeated = tf.reshape(self.x_repeated, [self.n_vimco_samples, -1, self.n_input])
        self.decoder_log_density_vimco = bernoulli_logit_density(self.x_repeated, self.x_reconst_vimco)
        self.decoder_log_density_vimco = tf.reduce_mean(self.decoder_log_density_vimco, axis=0)
        self.decoder_log_density_vimco = tf.stop_gradient(self.decoder_log_density_vimco)
        
        self.decoder_log_density_vimco = self.decoder_log_density - self.decoder_log_density_vimco
        self.vimco_cost = - self.encoder_log_density * self.decoder_log_density_vimco
        self.cost_for_encoder_weights = tf.reduce_mean(self.kl_divergency + self.vimco_cost)
        
        self._create_optimizer()
        
    def partial_fit(self, X, epoch=None):
        super().partial_fit(X)
        
class NVILVAE(VAE):
    def __init__(self, *args, baseline_learning_rate=1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_learning_rate = baseline_learning_rate
        
        self._create_network()
        
        self._create_loss_optimizer()
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
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
        
        self.nvil_cost = - self.encoder_log_density * (self.decoder_log_density - self.baseline)
        self.cost_for_encoder_weights = tf.reduce_mean(self.kl_divergency + self.nvil_cost)
        self._create_optimizer()
        
        self.cost_for_baseline = tf.reduce_mean((self.decoder_log_density - self.baseline)**2)
        self.baseline_optimizer = tf.train.AdamOptimizer(learning_rate=self.baseline_learning_rate)
        self.baseline_minimizer = self.decoder_optimizer.minimize(self.cost_for_baseline, 
                                                                 var_list=self.baseline_weights)
        
    def partial_fit(self, X, epoch=None):
        super().partial_fit(X)
        self.sess.run(self.baseline_minimizer, feed_dict={self.x: X, self.learning_rate: self.learning_rate_value})
        
    def save_weights(self, save_path):
        super().save_weights(save_path)
        self.baseline_saver.save(self.sess, save_path + '_baseline')
        
    def restore_weights(self, restore_path):
        super().restore_weights(restore_path)
        self.baseline_saver.restore(self.sess, restore_path + '_baseline')
        
class MuPropVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._create_network()
        
        self._create_loss_optimizer()
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
    def __str__(self):
        return 'MuPropVAE'
        
    def _create_network(self):
        self.z = tf.stop_gradient(self.z)
        self.x_reconst = self._create_decoder_part(self.z)
        if self.encoder_distribution == 'multinomial':
            self.z_mean = tf.reshape(self.z_mean, [tf.shape(self.x)[0], self.n_ary*self.n_z])
        
        self.x_reconst_mean = self._create_decoder_part(self.z_mean)
        self.decoder_log_density_mean = bernoulli_logit_density(self.x, self.x_reconst_mean)
        
        jacobian = tf.gradients(self.decoder_log_density_mean, self.z_mean)[0]
        self.decoder_log_density_mean = tf.stop_gradient(self.decoder_log_density_mean)
        jacobian = tf.stop_gradient(jacobian)
        
        self.linear_part = tf.stop_gradient(tf.reduce_sum(jacobian * (self.z - self.z_mean), axis=1))
        self.deterministic_term = tf.reduce_sum(jacobian * self.z_mean, axis=1)
    
    def _create_loss_optimizer(self):
        self._create_loss()
        
        self.decoder_log_density_adjusted = self.decoder_log_density - self.decoder_log_density_mean
        self.decoder_log_density_adjusted -= self.linear_part
        self.muprop_cost = - self.encoder_log_density * self.decoder_log_density_adjusted
        self.muprop_cost -= self.deterministic_term
        self.cost_for_encoder_weights = tf.reduce_mean(self.kl_divergency + self.muprop_cost)
        self._create_optimizer()
        
    def partial_fit(self, X, epoch=None):
        super().partial_fit(X)