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

def unroll_tensor(x):
    s = tf.shape(x)
    return tf.reshape(x, tf.stack([s[0] * s[1], s[2]])), s

def roll_tensor(x, sh):
    return tf.reshape(x, tf.stack([sh[0], sh[1], tf.shape(x)[1]]))

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

def init_weights(n_in, n_out, name, bias=None):
    w = tf.Variable(xavier_init(n_in, n_out), name=name)
    b = tf.Variable(tf.zeros([n_out]), name=name + '_b') if bias is None else tf.Variable(bias)
    return w, b

def build_layer(x, w, b, nonlinearity=None):
    y = tf.matmul(x, w) + b
    if nonlinearity:
        y = nonlinearity(y)
    return y

def compute_log_density(**params):
    if params['distribution'] == 'multinomial':
        x, logits = params['x'], params['logits']
        if len(params.values()) < 4:
            logp = -tf.nn.softplus(-logits)
            logip = -tf.nn.softplus(logits)
            return tf.reduce_sum(x * logp + (1. - x) * logip, -1)
        return tf.reduce_sum(-tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=logits), -1)
    elif params['distribution'] == 'gaussian':
        x, mu, sigma = params['x'], params['mu'], params['sigma']
        return -0.5 * tf.reduce_sum(((x - mu) / sigma) ** 2 + tf.log(2 * np.pi) + 2 * tf.log(sigma), 2)
    raise ValueError('Unsupported distribution!') 

def compute_kl_divergency(**params):
    if params['distribution'] == 'gaussian':
        mu, sigma = params['mu'], params['sigma']
        return -0.5 * tf.reduce_sum(1 + 2 * tf.log(sigma) - mu ** 2 - sigma ** 2, 2)
    if params['distribution'] == 'multinomial':
        q_probs = params['probs']
        q_logits = params['logits']
        prior_probs = params['prior_probs']
        kl = tf.reduce_sum(q_probs * (tf.nn.log_softmax(q_logits) - tf.log(prior_probs)), [-2, -1])
        return kl
    raise ValueError('Unsupported distribution!') 
    
def compute_multisample_elbo(log_density, kl_divergency, is_vimco=False):
    multisample_elbo = log_density - kl_divergency
    multisample_elbo = tf.transpose(multisample_elbo)
    max_w = tf.reduce_max(multisample_elbo, 0)
    adjusted_w = tf.exp(multisample_elbo - max_w)
    n_samples = tf.cast(tf.shape(adjusted_w)[0], tf.float32)
    if is_vimco is True:
        vimco_baseline = tf.reduce_mean(adjusted_w, 0) - adjusted_w / n_samples
        vimco_baseline = max_w + tf.log(1e-5 + vimco_baseline)
    adjusted_w = tf.reduce_mean((adjusted_w), 0)
    multisample_elbo = max_w + tf.log(adjusted_w)
    if is_vimco is True:
        multisample_elbo = multisample_elbo - tf.stop_gradient(vimco_baseline)
    return multisample_elbo

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
    learning_rate_step = 0
    steps = 0
    for epoch in tqdm(range(training_epochs)):
        epoch_train_loss = defaultdict(list)
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs, _ = X_train.next_batch(batch_size)
            batch_xs = batch_xs[:, None, :]
            for name, vae in zip(names, vaes):
                cost = vae.partial_fit(batch_xs)
                epoch_train_loss[name].append(cost)
        steps += 1
        if 3 ** learning_rate_step == steps:
            learning_rate_step += 1
            steps = 0

        if epoch % display_step == 0:
            clear_output()
            epoch_test_loss = defaultdict(list)
            test_batch_size = 50
            total_batch = int(X_test.num_examples / test_batch_size)
            for i in range(total_batch):
                batch_xs, _ = X_test.next_batch(batch_size)
                batch_xs = batch_xs[:, None, :]
                for name, vae in zip(names, vaes):
                    epoch_test_loss[name].append(vae.loss(batch_xs, n_samples=2))
            for name in names:
                test_loss[name].append(np.mean(epoch_test_loss[name]))
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