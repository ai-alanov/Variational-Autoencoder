import numpy as np
import tensorflow as tf
import collections
from itertools import chain
from tqdm import tqdm

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v[0]))
            items.append((new_key + '_b', v[1]))
    return dict(items)

def xavier_init(fan_in, fan_out, constant=1): 
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

def init_weights(n_in, n_out, name):
    w = tf.Variable(xavier_init(n_in, n_out), name=name)
    b = tf.Variable(tf.zeros([n_out]), name=name + '_b')
    return w, b

def build_layer(x, w, b, nonlinearity=None):
    y = tf.matmul(x, w) + b
    if nonlinearity:
        y = nonlinearity(y)
    return y

def log_normal_density(x, mu, sigma=1):
    return -0.5 * tf.reduce_sum(((x - mu) / sigma) ** 2 + tf.log(2 * np.pi) + 2 * tf.log(sigma), 1)

def bernoulli_logit_density(x, f):
    return tf.reduce_sum(x * tf.log(1e-8 + f) + (1. - x) * tf.log(1e-8 + 1 - f), -1)

def kl_divergency(mu, sigma):
    return -0.5 * tf.reduce_sum(1 + 2 * tf.log(sigma) - mu ** 2 - sigma ** 2, 1)

def get_gradient_mean_and_std(vae, batch_xs, n_iterations, gradient_type):
    gradients = []
    for _ in range(n_iterations):
        if gradient_type == 'decoder':
            gradient = vae.get_decoder_gradients(batch_xs)
        elif gradient_type == 'encoder':
            gradient = vae.get_encoder_gradients(batch_xs)
        gradients.append(gradient)
    gradients = np.array(gradients)
    gradient_std = np.linalg.norm(gradients - gradients.mean(axis=0)) / np.sqrt(n_iterations)
    return gradients.mean(axis=0), gradient_std

def train(vae, data, n_samples, batch_size, training_epochs=10, display_step=5, 
          weights_save_step=5, save_weights=True, save_path='saved_weights/'):
    test_loss = []
    for epoch in tqdm(range(training_epochs)):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs, _ = data.train.next_batch(batch_size)
            vae.partial_fit(batch_xs)
            cost = vae.loss(batch_xs)
            avg_cost += cost / n_samples * batch_size
        
        test_loss.append(vae.loss(data.test.images))
        if epoch % display_step == 0:
            print('Epoch: {:04d}, cost = {:.9f}, test cost = {:.9f}' \
                  .format(epoch+1, avg_cost, test_loss[-1]), flush=True)
        
        if epoch % weights_save_step == 0:
            if save_weights == True:
                vae.save_weights(save_path + '_{}'.format(epoch+1))
    test_loss = np.array(test_loss)
    vae.close()
    return test_loss