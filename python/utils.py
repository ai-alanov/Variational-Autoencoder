#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import collections
import os


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
    with tf.name_scope('reshape'):
        return tf.reshape(x, tf.stack([sh[0], sh[1], tf.shape(x)[1]]))


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_max = tf.reduce_max(y, 1, keep_dims=True)
        y_hard = tf.cast(tf.equal(y, y_max), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def init_weights(n_in, n_out, name, bias=None):
    w = tf.Variable(xavier_init(n_in, n_out), name='weights')
    bias = tf.zeros([n_out]) if bias is None else tf.Variable(bias)
    b = tf.Variable(bias, name='bias')
    return w, b


def build_layer(x, w, b, nonlinearity=None, name='layer'):
    with tf.name_scope(name):
        y = tf.matmul(x, w) + b
        if nonlinearity:
            y = nonlinearity(y)
    return y


def compute_log_density(**params):
    if 'name' not in params:
        params['name'] = 'compute_log_density'
    with tf.name_scope(params['name']):
        if params['distribution'] == 'multinomial':
            x, logits = params['x'], params['logits']
            if len(params.values()) < 5:
                logp = -tf.nn.softplus(-logits)
                logip = -tf.nn.softplus(logits)
                return tf.reduce_sum(x * logp + (1. - x) * logip, -1)
            entropy = -tf.nn.softmax_cross_entropy_with_logits(
                labels=x, logits=logits)
            return tf.reduce_sum(entropy, -1)
        elif params['distribution'] == 'gaussian':
            x, mu, sigma = params['x'], params['mu'], params['sigma']
            entropy = -0.5 * (((x - mu) / sigma) ** 2 + tf.log(2 * np.pi)
                              + 2 * tf.log(sigma))
            return tf.reduce_sum(entropy, 2)
        raise ValueError('Unsupported distribution!')


def compute_kl_divergency(**params):
    with tf.name_scope(params['name']):
        if params['distribution'] == 'gaussian':
            mu, sigma = params['mu'], params['sigma']
            kl = -0.5 * (1 + 2 * tf.log(sigma) - mu ** 2 - sigma ** 2)
            return tf.reduce_sum(kl, 2)
        if params['distribution'] == 'multinomial':
            q_samples = params['x']
            q_logits = params['logits']
            prior_probs = params['prior_probs']
            kl = tf.reduce_sum(q_samples * (tf.nn.log_softmax(q_logits) -
                                            tf.log(prior_probs)), [-2, -1])
            return kl
        raise ValueError('Unsupported distribution!')


def compute_multisample_elbo(log_density, kl_divergency, is_vimco=False,
                             name='multisample_elbo'):
    with tf.name_scope(name):
        multisample_elbo = log_density - kl_divergency
        multisample_elbo = tf.transpose(multisample_elbo)
        max_w = tf.reduce_max(multisample_elbo, 0)
        adjusted_w = tf.exp(multisample_elbo - max_w)
        n_samples = tf.cast(tf.shape(adjusted_w)[0], tf.float32)
        if is_vimco is True:
            vimco_baseline = tf.reduce_mean(adjusted_w, 0)
            vimco_baseline -= adjusted_w / n_samples
            vimco_baseline = max_w + tf.log(1e-5 + vimco_baseline)
        adjusted_w = tf.reduce_mean((adjusted_w), 0)
        multisample_elbo = max_w + tf.log(adjusted_w)
        if is_vimco is True:
            multisample_elbo -= tf.stop_gradient(vimco_baseline)
        return multisample_elbo


def get_gradient_mean_and_std(vae, sess, input_x, batch_xs, n_iterations,
                              gradient_type):
    gradients = []
    for _ in range(n_iterations):
        if gradient_type == 'decoder':
            gradient = vae.get_decoder_gradients(sess, input_x, batch_xs)
        elif gradient_type == 'encoder':
            gradient = vae.get_encoder_gradients(sess, input_x, batch_xs)
        gradients.append(gradient)
    gradients = np.array(gradients)
    gradient_std = np.linalg.norm(gradients - gradients.mean(axis=0))
    gradient_std /= np.sqrt(n_iterations)
    return gradients.mean(axis=0), gradient_std


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
