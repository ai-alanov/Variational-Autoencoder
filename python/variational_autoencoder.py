import numpy as np
import tensorflow as tf

def xavier_init(fan_in, fan_out, constant=1): 
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class VariationalAutoencoder(object):
    def __init__(self, n_input, n_z, network_architecture, 
                 learning_rate=0.001):
        self.n_input = n_input
        self.n_z = n_z
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        
        self.x = tf.placeholder(tf.float32, [None, n_input])
        
        self._create_network()
        
        self._create_loss_optimizer()
        
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        
    def _create_network(self):
        self.weights = self._initialize_weights(**self.network_architecture)
        
        encoder_layer1 = tf.nn.softplus(tf.add(tf.matmul(self.x, self.weights['encoder']['h1']),
                                               self.weights['encoder']['b1']))
        encoder_layer2 = tf.nn.softplus(tf.add(tf.matmul(encoder_layer1, self.weights['encoder']['h2']), 
                                               self.weights['encoder']['b2']))
        self.z_mean = tf.add(tf.matmul(encoder_layer2, self.weights['encoder']['out_mean']), 
                             self.weights['encoder']['out_mean_b'])
        self.z_log_sigma_sq = tf.add(tf.matmul(encoder_layer2, self.weights['encoder']['out_log_sigma_sq']), 
                                     self.weights['encoder']['out_log_sigma_sq_b'])
        epsilon = tf.random_normal((tf.shape(self.x)[0], self.n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), epsilon))
        
        decoder_layer1 = tf.nn.softplus(tf.add(tf.matmul(self.z, self.weights['decoder']['h1']),
                                               self.weights['decoder']['b1']))
        decoder_layer2 = tf.nn.softplus(tf.add(tf.matmul(decoder_layer1, self.weights['decoder']['h2']), 
                                               self.weights['decoder']['b2']))
        self.x_reconstruction = tf.sigmoid(tf.add(tf.matmul(decoder_layer2, self.weights['decoder']['out_mean']),
                                                  self.weights['decoder']['out_mean_b']))
        
    def _initialize_weights(self, n_hidden_encoder_1, n_hidden_encoder_2, 
                           n_hidden_decoder_1, n_hidden_decoder_2):
        weights = dict()
        weights['encoder'] = {
            'h1': tf.Variable(xavier_init(self.n_input, n_hidden_encoder_1)), 
            'h2': tf.Variable(xavier_init(n_hidden_encoder_1, n_hidden_encoder_2)), 
            'out_mean': tf.Variable(xavier_init(n_hidden_encoder_2, self.n_z)),
            'out_log_sigma_sq': tf.Variable(xavier_init(n_hidden_encoder_2, self.n_z)),
            
            'b1': tf.Variable(tf.zeros([n_hidden_encoder_1], dtype=tf.float32)), 
            'b2': tf.Variable(tf.zeros([n_hidden_encoder_2], dtype=tf.float32)), 
            'out_mean_b': tf.Variable(tf.zeros([self.n_z], dtype=tf.float32)),
            'out_log_sigma_sq_b': tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))
        }
        weights['decoder'] = {
            'h1': tf.Variable(xavier_init(self.n_z, n_hidden_decoder_1)), 
            'h2': tf.Variable(xavier_init(n_hidden_decoder_1, n_hidden_decoder_2)), 
            'out_mean': tf.Variable(xavier_init(n_hidden_encoder_2, self.n_input)),
            
            'b1': tf.Variable(tf.zeros([n_hidden_decoder_1], dtype=tf.float32)), 
            'b2': tf.Variable(tf.zeros([n_hidden_decoder_2], dtype=tf.float32)), 
            'out_mean_b': tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        }
        return weights
    
    def _create_loss_optimizer(self):
        decoder_loss = 0.5 * tf.reduce_sum(tf.square(tf.sub(self.x_reconstruction, self.x)))
        encoder_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                            - tf.square(self.z_mean) 
                                            - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(decoder_loss + encoder_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z=None):
        if z is None:
            z = np.random.normal(size=self.n_z)
        return self.sess.run(self.x_reconstruction, feed_dict={self.z: z})
    
    def reconstruct(self, X):
        return self.sess.run(self.x_reconstruction, feed_dict={self.x: X})
    
    def score(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})