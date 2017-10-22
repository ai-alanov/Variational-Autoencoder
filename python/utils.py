import numpy as np
import tensorflow as tf
import collections
import os
from itertools import chain
from tqdm import tqdm
from collections import defaultdict
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

from tensorflow.contrib.learn.python.learn.datasets import base, mnist
from tensorflow.python.framework import dtypes

import vae

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
    w = tf.Variable(xavier_init(n_in, n_out), name='weights')
    b = tf.Variable(tf.zeros([n_out]), name='bias') if bias is None else tf.Variable(bias)
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
            return tf.reduce_sum(-tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=logits), -1)
        elif params['distribution'] == 'gaussian':
            x, mu, sigma = params['x'], params['mu'], params['sigma']
            return -0.5 * tf.reduce_sum(((x - mu) / sigma) ** 2 + tf.log(2 * np.pi) + 2 * tf.log(sigma), 2)
        raise ValueError('Unsupported distribution!') 

def compute_kl_divergency(**params):
    with tf.name_scope(params['name']):
        if params['distribution'] == 'gaussian':
            mu, sigma = params['mu'], params['sigma']
            return -0.5 * tf.reduce_sum(1 + 2 * tf.log(sigma) - mu ** 2 - sigma ** 2, 2)
        if params['distribution'] == 'multinomial':
            q_samples = params['x']
            q_logits = params['logits']
            prior_probs = params['prior_probs']
            kl = tf.reduce_sum(q_samples * (tf.nn.log_softmax(q_logits) - tf.log(prior_probs)), [-2, -1])
            return kl
        raise ValueError('Unsupported distribution!') 
    
def compute_multisample_elbo(log_density, kl_divergency, is_vimco=False, name='multisample_elbo'):
    with tf.name_scope(name):
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

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def set_up_cuda_devices(cuda_devices):
    if cuda_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        
def set_up_vaes(vaes, common_params, learning_rates):
    input_x = tf.placeholder(tf.float32, [None, 1, common_params['n_input']])
    input_x = tf.cast(tf.random_uniform(tf.shape(input_x)) <= input_x, tf.float32)
    common_params['x'] = input_x
    vaes = [vae(**common_params, learning_rate=learning_rate) 
            for vae, learning_rate in zip(vaes, learning_rates)]
    return vaes, input_x

def set_up_session(vaes, config_params):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config_params['mem_fraction'])
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
#     for vae in vaes:
#         vae.initialize_weights(sess)
    return sess
    
def get_batch(data, batch_size):     #ToDo: get random batch
    batch_xs, _ = data.next_batch(batch_size)
    batch_xs = batch_xs[:, None, :]
    return batch_xs

def run_epoch(vaes, sess, input_x, data, n_samples, batch_size, obj_samples, 
              is_train=True, need_to_restore=False, save_path=None, epoch=None):
    costs = defaultdict(list)
    n_batches = int(n_samples / batch_size)
    for _ in range(n_batches):
        batch_xs = get_batch(data, batch_size)
        dict_of_tensors = {}
        feed_dict = {input_x: batch_xs}
        for vae in vaes:
            if is_train:
                vae_dict_of_tensors, vae_feed_dict = vae.partial_fit(n_samples=obj_samples)
            else:
                if need_to_restore:
                    vae.restore_weights(sess, os.path.join(save_path, vae.name() + '_{}'.format(epoch+1)))
                vae_dict_of_tensors, vae_feed_dict = vae.loss(n_samples=obj_samples)
            dict_of_tensors[vae.name()] = vae_dict_of_tensors
            feed_dict.update(vae_feed_dict)
        dict_of_results = sess.run(dict_of_tensors, feed_dict)
        for vae in vaes:
            costs[vae.name()].append(dict_of_results[vae.name()]['cost_for_display'])
    return dict([(vae.name(), np.mean(costs[vae.name()])) for vae in vaes])

def run_train_step(vaes, sess, input_x, train_params):
    return run_epoch(vaes, sess, input_x, **train_params)

def run_epoch_evaluation(vaes, sess, input_x, test_params, **kwargs):
    return run_epoch(vaes, sess, input_x, **test_params, is_train=False, **kwargs)

def print_costs(vaes, test_costs, val_costs=None, train_costs=None):
    for vae in vaes:
        name = vae.name()
        train_output = 'train cost = {:.9f}, '.format(train_costs[name]) if train_costs else ''
        val_output = 'validation cost = {:.9f}, '.format(val_costs[name]) if val_costs else ''
        test_output = 'test cost = {:.9f}'.format(test_costs[name])
        print(name + ': ' + train_output + val_output + test_output, flush=True)
        
def plot_loss(vaes, test_loss, val_loss, epoch, step, loss_name='Test+Validation', start=0):
    plt.figure(figsize=(12, 8))
    for vae in vaes:
        name = vae.name()
        plt.plot(np.arange(start, epoch+1, step), test_loss[name], label='Test ' + name)
        plt.plot(np.arange(start, epoch+1, step), val_loss[name], label='Validation ' + name)
    plt.title('{} loss'.format(loss_name))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()
    
def save_vae_weights(vaes, sess, epoch, save_path):
    makedirs(save_path)
    for vae in vaes:
        vae.save_weights(sess, os.path.join(save_path, vae.name() + '_{}'.format(epoch+1)))
        
def train_model(vaes, vae_params, train_params, val_params, test_params, config_params):
    set_up_cuda_devices(config_params['cuda_devices'])
    vaes, input_x = set_up_vaes(vaes, **vae_params)
    sess = set_up_session(vaes, config_params)
    val_loss = defaultdict(list)
    test_loss = defaultdict(list)
    for epoch in tqdm(range(config_params['n_epochs'])):
        train_costs = run_train_step(vaes, sess, input_x, train_params)
        if epoch % config_params['display_step'] == 0:
            val_costs = run_epoch_evaluation(vaes, sess, input_x, val_params)
            test_costs = run_epoch_evaluation(vaes, sess, input_x, test_params)
            for vae in vaes:
                val_loss[vae.name()].append(val_costs[vae.name()])
                test_loss[vae.name()].append(test_costs[vae.name()])
            clear_output()
            print_costs(vaes, test_costs, val_costs, train_costs)
            plot_loss(vaes, test_loss, val_loss, epoch, config_params['display_step'])
        if epoch % config_params['save_step'] == 0:
            if config_params['save_weights'] == True:
                save_vae_weights(vaes, sess, epoch, config_params['save_path'])
    for vae in vaes:
        vae.close()
    tf.reset_default_graph()
            
def test_model(vaes, vae_params, test_params, val_params, config_params):
    set_up_cuda_devices(config_params['cuda_devices'])
    vaes = set_up_vaes(vaes, **vae_params)
    test_loss = defaultdict(list)
    val_loss = defaultdict(list)
    for epoch in tqdm(range(config_params['save_step'], config_params['n_epochs'], config_params['save_step'])):
        test_costs = run_epoch_evaluation(vaes, test_params, need_to_restore=True, 
                                          save_path=config_params['save_path'], epoch=epoch)
        val_costs = run_epoch_evaluation(vaes, val_params, need_to_restore=True, 
                                         save_path=config_params['save_path'], epoch=epoch)
        for vae in vaes:
            test_loss[vae.name()].append(test_costs[vae.name()])
            val_loss[vae.name()].append(val_costs[vae.name()])
        clear_output()
        print_costs(vaes, test_costs, val_costs)
        plot_loss(vaes, test_loss, val_loss, epoch, config_params['save_step'], start=config_params['save_step'])
    makedirs(config_params['results_dir'])
    results_file = config_params['save_path'] + '-testm{}'.format(test_params['obj_samples'])
    with open(os.path.join(config_params['results_dir'], results_file), 'wb') as f:
        pickle.dump(test_loss, f)
    with open(os.path.join(config_params['results_dir'], results_file.replace('test', 'val')), 'wb') as f:
        pickle.dump(val_loss, f)
    for vae in vaes:
        vae.close()
    tf.reset_default_graph()

def calculate_stds(vaes, batch_xs, n_epochs, save_step, save_path, n_iterations, vae_part):
    stds = defaultdict(lambda: defaultdict(list))
    for weights_name in tqdm(map(lambda x: x.name(), vaes)):
        for vae in vaes:
            if vae.name() == 'NVILVAE' and weights_name != 'NVILVAE':
                continue
            for saved_index in range(1, n_epochs+1, save_step):
                vae.restore_weights(save_path + weights_name + '_{}'.format(saved_index))
                _, std = get_gradient_mean_and_std(vae, batch_xs, n_iterations, vae_part)
                stds[weights_name][vae.name()].append(std)
    return stds
    
def consider_stds(vaes, vae_params, data, n_epochs, batch_size, n_iterations, save_step, save_path, cuda_devices):
    set_up_cuda_devices('cuda_devices')
    vaes = set_up_vaes(vaes, **vae_params)
    batch_xs = get_batch(data, batch_size)
    
    encoder_stds = calculate_stds(vaes, batch_xs, n_epochs, save_step, save_path, n_iterations, 'encoder')
    decoder_stds = calculate_stds(vaes, batch_xs, n_epochs, save_step, save_path, n_iterations, 'decoder')
    
    for vae in vaes:
            vae.close()
    tf.reset_default_graph()
    
    n_vaes = len(vaes)
    weights_range = np.arange(1, n_epochs+1, save_step)

    fig, axes = plt.subplots(n_vaes, 2, figsize=(16, 40))
    for idx, weights_name in enumerate(map(lambda x: x.name(), vaes)):
        for name in map(lambda x: x.name(), vaes):
            if name == 'NVILVAE' and weights_name != 'NVILVAE':
                continue
            axes[idx][0].plot(weights_range, np.log10(encoder_stds[weights_name][name]), 
                              label='{} encoder log-std'.format(name))
            axes[idx][1].plot(weights_range, np.log10(decoder_stds[weights_name][name]), 
                              label='{} decoder log-std'.format(name))
        axes[idx][0].set_title('Encoder log-std, {} weights'.format(weights_name))
        axes[idx][0].set_xlabel('epoch')
        axes[idx][0].set_ylabel('log-std')
        axes[idx][0].legend(loc='best')

        axes[idx][1].set_title('Decoder log-std, {} weights'.format(weights_name))
        axes[idx][1].set_xlabel('epoch')
        axes[idx][1].set_ylabel('log-std')
        axes[idx][1].legend(loc='best')
    plt.show()

def binarized_mnist_fixed_binarization(datasets_dir, validation_size=0):
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    def shuffle(data):
        if data.shape[0] > 0:
            data = data[np.random.choice(data.shape[0], size=data.shape[0], replace=False)]
        return data
    with open(os.path.join(datasets_dir, 'BinaryMNIST', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(datasets_dir, 'BinaryMNIST', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(datasets_dir, 'BinaryMNIST', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')
    
    train_data = np.concatenate((validation_data, train_data))
    validation_data = train_data[:validation_size]
    train_data = train_data[validation_size:]
    
    train_data, validation_data, test_data = map(shuffle, [train_data, validation_data, test_data])
    train_data, validation_data, test_data = map(lambda x: 255.0 * x, [train_data, validation_data, test_data])

    options = dict(dtype=dtypes.float32, reshape=False, seed=1234)

    train = mnist.DataSet(train_data, train_data, **options)
    validation = mnist.DataSet(validation_data, validation_data, **options)
    test = mnist.DataSet(test_data, test_data, **options)

    return base.Datasets(train=train, validation=validation, test=test)

def choose_vaes_and_learning_rates(encoder_distribution, train_obj_samples, all_vaes=True):
    if encoder_distribution == 'gaussian':
        if all_vaes:
            vaes = [vae.VAE, vae.LogDerTrickVAE, vae.NVILVAE, vae.MuPropVAE]
        else:
            vaes = [vae.VAE, vae.LogDerTrickVAE, vae.NVILVAE, vae.MuPropVAE]
    elif encoder_distribution == 'multinomial':
        if all_vaes:
            vaes = [vae.LogDerTrickVAE, vae.NVILVAE, vae.MuPropVAE, vae.GumbelSoftmaxTrickVAE]
        else:
            vaes = [vae.LogDerTrickVAE, vae.NVILVAE, vae.MuPropVAE, vae.GumbelSoftmaxTrickVAE]
#             vaes = []
    if train_obj_samples > 1:
        vaes.append(vae.VIMCOVAE)
    learning_rates = [1e-4] * len(vaes)
    if encoder_distribution == 'gaussian':
        learning_rates[0] = 1e-3
    return vaes, learning_rates

def setup_input_vaes_and_params(X_train, X_val, X_test, binarized, n_z, n_ary, encoder_distribution,
                                train_obj_samples, val_batch_size, val_obj_samples, test_batch_size, 
                                test_obj_samples, cuda_devices, save_step, n_epochs=3001, save_weights=True, 
                                mem_fraction=0.333, all_vaes=True, mode='train', results_dir=None):
    n_input = X_train.images.shape[1]
    n_hidden = 200
    network_architecture = {
        'encoder': {
            'h1': (n_input, n_hidden), 
            'h2': (n_hidden, n_ary*n_hidden), 
            'out_mean': (n_ary*n_hidden, n_ary*n_z)
        }, 
        'decoder': {
            'h1': (n_ary*n_z, n_hidden), 
            'h2': (n_hidden, n_hidden),
            'out_mean': (n_hidden, n_input)
        }
    }
    if encoder_distribution == 'gaussian':
        network_architecture['encoder']['out_log_sigma_sq'] = (n_ary*n_hidden, n_ary*n_z)
        
    train_params = {
        'data': X_train,
        'n_samples': X_train.num_examples,
        'batch_size': 128,
        'obj_samples': train_obj_samples
    }
    val_params = {
        'data': X_val,
        'n_samples': X_val.num_examples,
        'batch_size': val_batch_size,
        'obj_samples': val_obj_samples
    }
    test_params = {
        'data': X_test,
        'n_samples': X_test.num_examples,
        'batch_size': test_batch_size, 
        'obj_samples': test_obj_samples
    }
    config_params = {
        'n_epochs': n_epochs,
        'display_step': 100,
        'save_weights': save_weights,
        'save_step': save_step,
        'save_path': '{}VAE-{}B-m{}-nz{}'.format('Bin-' if binarized else '', n_ary, train_obj_samples, n_z),
        'cuda_devices': cuda_devices,
        'mem_fraction': mem_fraction,
        'results_dir': results_dir
    }
    vae_params = {
        'common_params': {
            'n_input': n_input,
            'n_z': n_z,
            'n_ary': n_ary,
            'n_samples': train_params['obj_samples'],
            'encoder_distribution': encoder_distribution,
            'network_architecture': network_architecture
        }
    }

    vaes, learning_rates = choose_vaes_and_learning_rates(encoder_distribution, train_obj_samples, all_vaes=all_vaes)
    vae_params['learning_rates'] = learning_rates
    
    input_vaes_and_params = {
        'vaes': vaes, 
        'vae_params': vae_params,
        'train_params': train_params,
        'val_params': val_params,
        'test_params': test_params,
        'config_params': config_params
    }
    if mode == 'test':
        input_vaes_and_params.pop('train_params')
    return input_vaes_and_params