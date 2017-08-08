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

def set_up_cuda_devices(cuda_devices):
    if cuda_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        
def set_up_vaes(vaes, common_params, learning_rates):
    vaes = [vae(**common_params, learning_rate=learning_rate) 
            for vae, learning_rate in zip(vaes, learning_rates)]
    return vaes

def run_epoch(vaes, data, n_samples, batch_size, obj_samples, is_train=True,
              need_to_restore=False, save_path=None, epoch=None):
    costs = defaultdict(list)
    n_batches = int(n_samples / batch_size)
    for _ in range(n_batches):
        batch_xs, _ = data.next_batch(batch_size)
        batch_xs = batch_xs[:, None, :]
        for vae in vaes:
            if is_train:
                cost = vae.partial_fit(batch_xs, n_samples=obj_samples)
            else:
                if need_to_restore:
                    vae.restore_weights(save_path + vae.name() + '_{}'.format(epoch+1))
                cost = vae.loss(batch_xs, n_samples=obj_samples)
            costs[vae.name()].append(cost)
    return dict([(vae.name(), np.mean(costs[vae.name()])) for vae in vaes])

def run_train_step(vaes, train_params):
    return run_epoch(vaes, **train_params)

def run_epoch_evaluation(vaes, test_params, **kwargs):
    return run_epoch(vaes, **test_params, is_train=False, **kwargs)

def print_costs(vaes, test_costs, train_costs=None):
    for vae in vaes:
        name = vae.name()
        train_output = 'train cost = {:.9f}, '.format(train_costs[name]) if train_costs else ''
        test_output = 'test cost = {:.9f}'.format(test_costs[name])
        print(name + ': ' + train_output + test_output, flush=True)
        
def plot_loss(vaes, loss, epoch, step, loss_name='Test', start=0):
    plt.figure(figsize=(12, 8))
    for vae in vaes:
        name = vae.name()
        plt.plot(np.arange(start, epoch+1, step), loss[name], label=name)
    plt.title('{} loss'.format(loss_name))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()
    
def save_vae_weights(vaes, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for vae in vaes:
        vae.save_weights(save_path + vae.name() + '_{}'.format(epoch+1))
        
def train_model(vaes, vae_params, train_params, test_params, config_params):
    set_up_cuda_devices(config_params['cuda_devices'])
    vaes = set_up_vaes(vaes, **vae_params)
    test_loss = defaultdict(list)
    for epoch in tqdm(range(config_params['n_epochs'])):
        train_costs = run_train_step(vaes, train_params)
        if epoch % config_params['display_step'] == 0:
            test_costs = run_epoch_evaluation(vaes, test_params)
            for vae in vaes:
                test_loss[vae.name()].append(test_costs[vae.name()])
            clear_output()
            print_costs(vaes, test_costs, train_costs)
            plot_loss(vaes, test_loss, epoch, config_params['display_step'])
        if epoch % config_params['save_step'] == 0:
            if config_params['save_weights'] == True:
                save_vae_weights(vaes, epoch, config_params['save_path'])
    for vae in vaes:
        vae.close()
    tf.reset_default_graph()
            
def test_model(vaes, vae_params, test_params, config_params):
    set_up_cuda_devices(config_params['cuda_devices'])
    vaes = set_up_vaes(vaes, **vae_params)
    test_loss = defaultdict(list)
    for epoch in tqdm(range(config_params['save_step'], config_params['n_epochs'], config_params['save_step'])):
        test_costs = run_epoch_evaluation(vaes, test_params, need_to_restore=True, 
                                          save_path=config_params['save_path'], epoch=epoch)
        for vae in vaes:
            test_loss[vae.name()].append(test_costs[vae.name()])
        clear_output()
        print_costs(vaes, test_costs)
        plot_loss(vaes, test_loss, epoch, config_params['save_step'], start=config_params['save_step'])
    for vae in vaes:
        vae.close()
    tf.reset_default_graph()
    
def get_batch(data, batch_size):     #ToDo: get random batch
    batch_xs, _ = data.next_batch(batch_size)
    batch_xs = batch_xs[:, None, :]
    return batch_xs

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
    

# def train(vaes, names, X_train, X_test, train_batch_size, test_batch_size, train_obj_samples, test_obj_samples,
#           training_epochs, display_step, weights_save_step, 
#           save_weights=True, save_path='saved_weights/', cuda_devices=None):
#     if cuda_devices is not None:
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
#     test_loss = defaultdict(list)
#     n_train_samples = X_train.num_examples
#     n_test_samples = X_test.num_examples
#     for epoch in tqdm(range(training_epochs)):
#         epoch_train_loss = defaultdict(list)
#         total_batch = int(n_train_samples / train_batch_size)
#         for i in range(total_batch):
#             batch_xs, _ = X_train.next_batch(train_batch_size)
#             batch_xs = batch_xs[:, None, :]
#             for name, vae in zip(names, vaes):
#                 cost = vae.partial_fit(batch_xs, n_samples=train_obj_samples)
#                 epoch_train_loss[name].append(cost)

#         if epoch % display_step == 0:
#             clear_output()
#             epoch_test_loss = defaultdict(list)
#             total_batch = int(n_test_samples / test_batch_size)
#             for i in range(total_batch):
#                 batch_xs, _ = X_test.next_batch(test_batch_size)
#                 batch_xs = batch_xs[:, None, :]
#                 for name, vae in zip(names, vaes):
#                     epoch_test_loss[name].append(vae.loss(batch_xs, n_samples=test_obj_samples))
#             for name in names:
#                 test_loss[name].append(np.mean(epoch_test_loss[name]))
#                 print('{0}: train cost = {1:.9f}, test cost = {2:.9f}'.format(name, np.mean(epoch_train_loss[name]), 
#                                                                               test_loss[name][-1]), flush=True)
#             plt.figure(figsize=(12, 8))
#             for name in names:
#                 plt.plot(np.arange(0, epoch+1, display_step), test_loss[name], label=name)
#             plt.title('Test loss')
#             plt.xlabel('epoch')
#             plt.ylabel('loss')
#             plt.legend(loc='best')
#             plt.show()

#         if epoch % weights_save_step == 0:
#             if save_weights == True:
#                 if not os.path.exists(save_path):
#                     os.makedirs(save_path)
#                 for name, vae in zip(names, vaes):
#                     vae.save_weights(save_path + name + '_{}'.format(epoch+1))
#     for vae in vaes:
#         vae.close()
#     tf.reset_default_graph()
    
# def test(vaes, names, X_test, test_batch_size, test_obj_samples,
#           training_epochs, display_step, weights_save_step, 
#           save_path='saved_weights/', cuda_devices=None):
#     if cuda_devices is not None:
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
#     test_loss = defaultdict(list)
#     n_test_samples = X_test.num_examples
#     for test_epoch in tqdm(range(0, training_epochs, weights_save_step)):
#         epoch_test_loss = defaultdict(list)
#         total_batch = int(n_test_samples / test_batch_size)
#         for i in range(total_batch):
#             batch_xs, _ = X_test.next_batch(test_batch_size)
#             batch_xs = batch_xs[:, None, :]
#             for name, vae in zip(names, vaes):
#                 vae.restore_weights(save_path + name + '_{}'.format(test_epoch+1))
#                 epoch_test_loss[name].append(vae.loss(batch_xs, n_samples=test_obj_samples))
#         clear_output()
#         for name in names:
#             test_loss[name].append(np.mean(epoch_test_loss[name]))
#             print('{0}: train cost = {1:.9f}, test cost = {2:.9f}'.format(name, np.mean(epoch_test_loss[name]), 
#                                                                           test_loss[name][-1]), flush=True)
#         plt.figure(figsize=(12, 8))
#         for name in names:
#             plt.plot(np.arange(0, test_epoch+1, weights_save_step), test_loss[name], label=name)
#         plt.title('Test loss')
#         plt.xlabel('epoch')
#         plt.ylabel('loss')
#         plt.legend(loc='best')
#         plt.show()

#     for vae in vaes:
#         vae.close()
#     tf.reset_default_graph()