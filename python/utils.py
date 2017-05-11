import numpy as np
import tensorflow as tf
import collections
import os
from itertools import chain
from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

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

def sample_gumbel(shape, eps=1e-20): 
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

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
    f *= 1 - 2e-7
    f += 1e-7
    return tf.reduce_sum(x * tf.log(f) + (1. - x) * tf.log(1 - f), -1)

def log_density(params, distribution='multinomial'):
    if distribution == 'multinomial':
        x, probs = params['x'], params['probs']
        if len(params.values()) < 3:
            return tf.reduce_sum(x * tf.log(1e-8 + probs) + (1. - x) * tf.log(1e-8 + 1 - probs), -1)
        return tf.reduce_sum(x * tf.log(1e-8 + probs), 1)
    elif distribution == 'gaussian':
        x, mu, sigma = params['x'], params['mu'], params['sigma']
        return -0.5 * tf.reduce_sum(((x - mu) / sigma) ** 2 + tf.log(2 * np.pi) + 2 * tf.log(sigma), 1)
    raise ValueError('Unsupported distribution!') 

def kl_divergency(params, distribution='multinomial'):
    if distribution == 'gaussian':
        mu, sigma = params['mu'], params['sigma']
        return -0.5 * tf.reduce_sum(1 + 2 * tf.log(sigma) - mu ** 2 - sigma ** 2, 1)
    if distribution == 'multinomial':
        q_probs = params['probs']
        prior_probs = params['prior_probs']
        return tf.reduce_sum(q_probs * (tf.log(1e-8 + q_probs) - tf.log(prior_probs)), 1)
    raise ValueError('Unsupported distribution!') 

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

def train(vaes, names, X_train, X_test, n_samples, batch_size, training_epochs, display_step, 
          weights_save_step, save_weights=True, save_path='saved_weights/'):
    test_loss = defaultdict(list)
    for epoch in tqdm(range(training_epochs)):
        epoch_train_loss = defaultdict(list)
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs, _ = X_train.next_batch(batch_size)
            for name, vae in zip(names, vaes):
                vae.partial_fit(batch_xs, epoch=epoch+1)
                cost = vae.loss(batch_xs)
                epoch_train_loss[name].append(cost)
        for name, vae in zip(names, vaes):
            test_loss[name].append(vae.loss(X_test.images))

        if epoch % display_step == 0:
            clear_output()
            for name in names:
                print('{0}: train cost = {1:.9f}, test cost = {2:.9f}'.format(name, np.mean(epoch_train_loss[name]), 
                                                                              test_loss[name][-1]), flush=True)
            plt.figure(figsize=(12, 8))
            for name in names:
                plt.plot(test_loss[name], label=name)
            plt.title('Test loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc='best')
            plt.show()

        if epoch % weights_save_step == 0:
            if save_weights == True:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for name, vae in zip(names, vaes):
                    vae.save_weights(save_path + name + '_{}'.format(epoch+1))
    for vae in vaes:
        vae.close()
    tf.reset_default_graph()