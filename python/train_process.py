#!/usr/bin/env python3
import numpy as np
import scipy.io
import tensorflow as tf
import os
import sys
import glob
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import pickle

import matplotlib.pyplot as plt
import seaborn as sns  # noqa
from IPython import display

from tensorflow.contrib.learn.python.learn.datasets import base, mnist
from tensorflow.python.framework import dtypes

from utils import makedirs, get_gradient_mean_and_std
from vae import *  # noqa


def set_up_cuda_devices(cuda_devices):
    if cuda_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices


def set_up_vaes(vaes, vae_params):
    input_x = tf.placeholder(tf.float32, [None, 1, vae_params['n_input']])
    input_x = tf.random_uniform(tf.shape(input_x)) <= input_x
    input_x = tf.cast(input_x, tf.float32)
    vae_params['x'] = input_x
    vaes = [vae(**vae_params) for vae in vaes]
    return vaes, input_x


def set_up_session(vaes, config_params):
    mem_frac = config_params['mem_fraction']
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
#     for vae in vaes:
#         vae.initialize_weights(sess)
    return sess


def get_batch(data, batch_size):  # ToDo: get random batch
    batch_xs, _ = data.next_batch(batch_size)
    batch_xs = batch_xs[:, None, :]
    return batch_xs


def clear_output():
    if sys.stdout.isatty():
        pass  # os.system('cls' if os.name == 'nt' else 'clear')
    else:
        display.clear_output()


def log_output(output, logging_path=None, **kwargs):
    if sys.stdout.isatty() or (logging_path is not None):
        with open(logging_path, 'a') as f:
            print(output, file=f, **kwargs)
    else:
        print(output, **kwargs)


def print_costs(vaes, epoch, test_costs, val_costs=None, train_costs=None,
                logging_path=None):
    log_output(epoch + 1, logging_path)
    for vae in vaes:
        vae_name = vae.name()
        data_names = ['test', 'validation', 'train']
        costs = {
            'test': test_costs[vae_name],
            'validation': val_costs[vae_name] if val_costs else None,
            'train': train_costs[vae_name] if train_costs else None
        }
        all_output = vae_name + ': '
        for name in data_names:
            output = '{} cost = {:.5f} '
            output = output.format(name, costs[name]) if costs[name] else ''
            all_output += output
        log_output(all_output, logging_path, flush=True)


def plot_loss(vaes, test_loss, val_loss, epoch, step,
              loss_name='Test+Validation', start=0):
    plt.figure(figsize=(12, 8))
    for vae in vaes:
        name = vae.name()
        plt.plot(np.arange(start, epoch + 1, step), test_loss[name],
                 label='Test ' + name)
        plt.plot(np.arange(start, epoch + 1, step), val_loss[name],
                 label='Validation ' + name)
    plt.title('{} loss'.format(loss_name))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()


def save_vae_weights(vaes, sess, epoch, save_dir):
    for vae in vaes:
        save_path = os.path.join(save_dir, vae.name(), vae.dataset_name(),
                                 vae.parameters())
        if epoch == 0:
            now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        else:
            latest_file = sorted(glob.glob(os.path.join(save_path, '*')))[-1]
            now = os.path.basename(latest_file)
        save_path = os.path.join(save_path, now)
        makedirs(save_path)
        vae.save_weights(sess, os.path.join(save_path, '{}'.format(epoch + 1)))


def find_file(vae, save_dir, lr):
    save_path = os.path.join(save_dir, vae.name(), vae.dataset_name(),
                             vae.parameters(learning_rate=lr))
    files = glob.glob(os.path.join(save_path, '*'))
    if not files:
        save_path = save_path[:-1]
        files = glob.glob(os.path.join(save_path, '*'))
    latest_file = sorted(files)[-1]
    now = os.path.basename(latest_file)
    return save_path, now


def restore_weights_file(vae, epoch, save_dir, learning_rate):
    if isinstance(learning_rate, float):
        val_lr = learning_rate
    elif isinstance(learning_rate, list):
        val_lr = learning_rate[0]
    elif isinstance(learning_rate, dict):
        val_lr = learning_rate[vae.name()]
    else:
        error_message = 'Unsupported type of learning_rate: {}'
        error_message = error_message.format(type(learning_rate))
        raise ValueError(error_message)
    save_path, now = find_file(vae, save_dir, val_lr)
    save_path = os.path.join(save_path, now)
    vae_epoch = epoch if isinstance(epoch, int) else epoch[vae.name()]
    weights_file = os.path.join(save_path, '{}'.format(vae_epoch + 1))
    return weights_file


def restore_vae_weights(vaes, sess, epoch, save_dir, learning_rate):
    for vae in vaes:
        weights_file = restore_weights_file(vae, epoch, save_dir,
                                            learning_rate)
        vae.restore_weights(sess, weights_file)


def load_loss(vaes, learning_rates, save_dir, results_dir, loss_name):
    loss = defaultdict(lambda: defaultdict(list))
    for lr in learning_rates:
        for vae in vaes:
            save_path, now = find_file(vae, save_dir, lr)
            save_path = os.path.join(save_path, now, results_dir)
            file_name = os.path.join(save_path, loss_name)
            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    loss[str(lr)].update(pickle.load(f))
    return loss


def save_loss(vaes, loss, save_dir, results_dir, loss_name,
              epochs=None, learning_rates=None):
    if 'test_samples' in loss:
        file_name = str(loss['test_samples']) + '-' + vaes[0].dataset_name()
        file_name += '-' + vaes[0].parameters(without_lr=True)
        other_save_path = os.path.join(
            results_dir, file_name)
        with open(other_save_path, 'wb') as f:
            pickle.dump(loss, f)
        for vae in vaes:
            lr = learning_rates[vae.name()]
            save_path, now = find_file(vae, save_dir, lr)
            save_path = os.path.join(save_path, now, results_dir)
            makedirs(save_path)
            file_name = now + '_' + loss_name
            file_name = os.path.join(save_path, file_name)
            with open(file_name, 'wb') as f:
                info = {
                    vae.name(): loss[vae.name()],
                    'best_epoch': epochs[vae.name()]
                }
                pickle.dump(info, f)
    else:
        for lr in list(loss.keys()):
            for vae in vaes:
                save_path, now = find_file(vae, save_dir, float(lr))
                save_path = os.path.join(save_path, now, results_dir)
                makedirs(save_path)
                file_name = loss_name
                file_name = os.path.join(save_path, file_name)
                with open(file_name, 'wb') as f:
                    info = {
                        vae.name(): loss[lr][vae.name()]
                    }
                    pickle.dump(info, f)


def run_epoch(vaes, sess, input_x, data, n_samples, batch_size,
              obj_samples, is_train=True, need_to_restore=False,
              save_path=None, epoch=None, learning_rate=None):
    costs = defaultdict(list)
    n_batches = int(n_samples / batch_size)
    if need_to_restore:
        restore_vae_weights(vaes, sess, epoch, save_path, learning_rate)
    lr_decay = 10 ** (-(epoch // 150) / 2) if epoch else 1.0
    if epoch and epoch % 150 == 0:
        print('lr: {:.10f}'.format(vaes[0].learning_rate_value * lr_decay))
    for _ in range(n_batches):
        batch_xs = get_batch(data, batch_size)
        dict_of_tensors = {}
        feed_dict = {input_x: batch_xs}
        for vae in vaes:
            if is_train:
                d_tensors, f_dict = vae.partial_fit(
                    n_samples=obj_samples, is_train=is_train,
                    learning_rate_decay=lr_decay)
            else:
                d_tensors, f_dict = vae.loss(n_samples=obj_samples,
                                             is_train=is_train)
            dict_of_tensors[vae.name()] = d_tensors
            feed_dict.update(f_dict)
        dict_of_results = sess.run(dict_of_tensors, feed_dict)
        for vae in vaes:
            name = vae.name()
            costs[name].append(dict_of_results[name]['cost_for_display'])
    return dict([(vae.name(), np.mean(costs[vae.name()])) for vae in vaes])


def run_train_step(vaes, sess, input_x, train_params, **kwargs):
    return run_epoch(vaes, sess, input_x, **train_params, **kwargs)


def run_epoch_evaluation(vaes, sess, input_x, test_params, **kwargs):
    return run_epoch(vaes, sess, input_x, **test_params,
                     is_train=False, **kwargs)


def train_model(vaes, vae_params, train_params, val_params, test_params,
                config_params):
    set_up_cuda_devices(config_params['cuda_devices'])
    vaes, input_x = set_up_vaes(vaes, vae_params)
    sess = set_up_session(vaes, config_params)
    val_loss = defaultdict(list)
    test_loss = defaultdict(list)
    for epoch in tqdm(range(config_params['n_epochs'])):
        train_costs = run_train_step(vaes, sess, input_x,
                                     train_params, epoch=epoch)
        if epoch % config_params['display_step'] == 0:
            val_costs = run_epoch_evaluation(vaes, sess, input_x, val_params)
            test_costs = run_epoch_evaluation(vaes, sess, input_x, test_params)
            for vae in vaes:
                val_loss[vae.name()].append(val_costs[vae.name()])
                test_loss[vae.name()].append(test_costs[vae.name()])
            clear_output()
            print_costs(vaes, epoch, test_costs, val_costs,
                        train_costs, config_params['logging_path'])
            plot_loss(vaes, test_loss, val_loss, epoch,
                      config_params['display_step'])
        if epoch % config_params['save_step'] == 0:
            if config_params['save_weights']:
                save_vae_weights(vaes, sess, epoch, config_params['save_path'])
    sess.close()
    tf.reset_default_graph()


def grid_search_on_validation(sess, vaes, input_x, val_params, config_params):
    learning_rates = config_params['learning_rates']
    save_path = config_params['save_path']
    val_loss = load_loss(vaes, learning_rates, save_path,
                         config_params['results_dir'], 'val.pkl')
    noncashed_vaes = defaultdict(list)
    for lr in learning_rates:
        noncashed_vaes[str(lr)] = [vae for vae in vaes
                                   if not val_loss[str(lr)][vae.name()]]
    for lr in learning_rates:
        for epoch in tqdm(range(0, config_params['n_epochs'],
                                config_params['save_step'])):
            val_costs = run_epoch_evaluation(noncashed_vaes[str(lr)], sess,
                                             input_x, val_params,
                                             need_to_restore=True,
                                             save_path=save_path,
                                             epoch=epoch, learning_rate=lr)
            for vae in noncashed_vaes[str(lr)]:
                val_loss[str(lr)][vae.name()].append(val_costs[vae.name()])
    save_loss(vaes, val_loss, save_path,
              config_params['results_dir'], 'val.pkl')
    min_loss = defaultdict(lambda: (np.inf, (None, None)))
    for lr in val_loss.keys():
        for vae in vaes:
            n_steps = np.nanargmin(val_loss[str(lr)][vae.name()])
            min_loss_value = val_loss[str(lr)][vae.name()][n_steps]
            n_steps *= config_params['save_step']
            if min_loss_value < min_loss[vae.name()][0]:
                min_loss[vae.name()] = (min_loss_value, (n_steps, float(lr)))
    optimal_epochs = {vae.name(): min_loss[vae.name()][1][0] for vae in vaes}
    optimal_lrs = {vae.name(): min_loss[vae.name()][1][1] for vae in vaes}
    return optimal_epochs, optimal_lrs


def test_model(vaes, vae_params, test_params, val_params, config_params):
    set_up_cuda_devices(config_params['cuda_devices'])
    vaes, input_x = set_up_vaes(vaes, vae_params)
    sess = set_up_session(vaes, config_params)
    epochs, learning_rates = grid_search_on_validation(
        sess, vaes, input_x, val_params, config_params)
    save_path = config_params['save_path']
    test_loss = run_epoch_evaluation(
        vaes, sess, input_x, test_params, need_to_restore=True,
        save_path=save_path, epoch=epochs, learning_rate=learning_rates)
    test_loss = {
        name: (test_loss[name], (epochs[name], learning_rates[name]))
        for name in map(lambda x: x.name(), vaes)
    }
    test_loss['test_samples'] = test_params['obj_samples']
    save_loss(vaes, test_loss, save_path, config_params['results_dir'],
              'Test-m{}.pkl'.format(test_params['obj_samples']),
              epochs, learning_rates)
    sess.close()
    tf.reset_default_graph()


def calculate_stds(vaes, batch_xs, n_epochs, save_step, save_path,
                   n_iterations, vae_part):
    stds = defaultdict(lambda: defaultdict(list))
    for weights_name in tqdm(map(lambda x: x.name(), vaes)):
        for vae in vaes:
            if vae.name() == 'NVILVAE' and weights_name != 'NVILVAE':
                continue
            for saved_index in range(1, n_epochs + 1, save_step):
                vae.restore_weights(save_path + weights_name
                                    + '_{}'.format(saved_index))
                _, std = get_gradient_mean_and_std(vae, batch_xs,
                                                   n_iterations, vae_part)
                stds[weights_name][vae.name()].append(std)
    return stds


def calculate_stds2(vaes, sess, input_x, batch_xs, config_params,
                    vae_part, weights):
    stds = defaultdict(list)
    for vae in tqdm(vaes):
        for saved_index in tqdm(range(0, config_params['n_epochs'],
                                      config_params['save_step'])):
            weights_file = restore_weights_file(
                vae, saved_index,
                config_params['save_path'],
                config_params['learning_rates'])
            vae.restore_weights(sess, weights_file)
            _, std = get_gradient_mean_and_std(
                vae, sess, input_x, batch_xs,
                config_params['n_iterations'], vae_part)
            stds[vae.name()].append(min(1e5, std))
    return stds


def plot_stds(vaes, vae_params, train_params, config_params):
    set_up_cuda_devices(config_params['cuda_devices'])
    vaes, input_x = set_up_vaes(vaes, vae_params)
    sess = set_up_session(vaes, config_params)
    batch_xs = get_batch(train_params['data'], train_params['batch_size'])
    all_weights = {vae.name(): vae for vae in vaes}
    weights = config_params['weights']
    weights = {w: all_weights[w] for w in weights} if weights else all_weights

    encoder_stds = calculate_stds2(vaes, sess, input_x, batch_xs,
                                   config_params, 'encoder', weights)

    sess.close()
    tf.reset_default_graph()

    n_weights = len(weights)
    weights_range = np.arange(1, config_params['n_epochs'] + 1,
                              config_params['save_step'])
    names = {
        'ReparamTrickVAE': 'RGrad',
        'VIMCOVAE': 'VIMCO',
        'LogDerTrickVAE': 'ScoreFunc',
        'NVILVAE': 'NVIL',
        'MuPropVAE': 'MuProp',
        'GumbelSoftmaxTrickVAE': 'GSoft'
    }
    fig, axes = plt.subplots(n_weights, 1, figsize=(8, 6))
    for idx, name in enumerate(map(lambda x: x.name(), vaes)):
        ax = axes[idx][0] if n_weights > 1 else axes
        ax.plot(weights_range,
                np.log10(encoder_stds[name]),
                label=names[name])
    ax.set_xlabel('epoch')
    ax.set_ylabel('log-std')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('test.pdf')
    plt.show()


def consider_stds(vaes, vae_params, data, n_epochs, batch_size, n_iterations,
                  save_step, save_path, cuda_devices):
    set_up_cuda_devices('cuda_devices')
    vaes = set_up_vaes(vaes, vae_params)
    batch_xs = get_batch(data, batch_size)

    encoder_stds = calculate_stds(vaes, batch_xs, n_epochs, save_step,
                                  save_path, n_iterations, 'encoder')
    decoder_stds = calculate_stds(vaes, batch_xs, n_epochs, save_step,
                                  save_path, n_iterations, 'decoder')

    for vae in vaes:
        vae.close()
    tf.reset_default_graph()

    n_vaes = len(vaes)
    weights_range = np.arange(1, n_epochs + 1, save_step)

    fig, axes = plt.subplots(n_vaes, 2, figsize=(16, 40))
    for idx, weights_name in enumerate(map(lambda x: x.name(), vaes)):
        for name in map(lambda x: x.name(), vaes):
            if name == 'NVILVAE' and weights_name != 'NVILVAE':
                continue
            axes[idx][0].plot(weights_range,
                              np.log10(encoder_stds[weights_name][name]),
                              label='{} encoder log-std'.format(name))
            axes[idx][1].plot(weights_range,
                              np.log10(decoder_stds[weights_name][name]),
                              label='{} decoder log-std'.format(name))
        title = '{} log-std, {} weights'
        axes[idx][0].set_title(title.format('Encoder', weights_name))
        axes[idx][0].set_xlabel('epoch')
        axes[idx][0].set_ylabel('log-std')
        axes[idx][0].legend(loc='best')

        axes[idx][1].set_title(title.format('Decoder', weights_name))
        axes[idx][1].set_xlabel('epoch')
        axes[idx][1].set_ylabel('log-std')
        axes[idx][1].legend(loc='best')
    plt.show()


def shuffle(data):
    if data.shape[0] > 0:
        data = data[np.random.choice(data.shape[0],
                                     size=data.shape[0],
                                     replace=False)]
    return data


def download_mnist(datasets_dir):
    import urllib.request
    import gzip

    mnist_filenames = ['train-images-idx3-ubyte', 't10k-images-idx3-ubyte']
    for filename in mnist_filenames:
        path_to_file = os.path.join(datasets_dir, "MNIST", filename)
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        url = "http://yann.lecun.com/exdb/mnist/{}.gz".format(filename)
        urllib.request.urlretrieve(url, path_to_file + '.gz')
        with gzip.open(path_to_file + '.gz', 'rb') as f:
            file_content = f.read()
        with open(path_to_file, 'wb') as f:
            f.write(file_content)
        os.remove(path_to_file + '.gz')

    subdatasets = ['train', 'valid', 'test']
    for subdataset in subdatasets:
        filename = 'binarized_mnist_{}.amat'.format(subdataset)
        url = 'http://www.cs.toronto.edu/~larocheh/public/datasets'
        url = os.path.join(url, 'binarized_mnist', 'binarized_mnist_{}.amat')
        url = url.format(subdataset)
        path_to_file = os.path.join(datasets_dir, "BinaryMNIST", filename)
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        urllib.request.urlretrieve(url, path_to_file)


def download_omniglot(datasets_dir, binarize=True):
    import urllib.request

    filename = 'chardata.mat'
    url = 'https://raw.github.com/yburda/iwae/master/datasets/OMNIGLOT'
    url = os.path.join(url, filename)
    path_to_file = os.path.join(datasets_dir, "OMNIGLOT", filename)
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    if not os.path.exists(path_to_file):
        urllib.request.urlretrieve(url, path_to_file)

    omni_raw = scipy.io.loadmat(path_to_file)
    datasets = [omni_raw['data'], omni_raw['testdata']]
    datasets = list(map(lambda x: x.astype('float32'), datasets))

    def binarize(data):
        return (np.random.rand(*data.shape) <= data).astype('float32')
    if binarize:
        datasets = list(map(lambda x: binarize(x), datasets))
    omni_raw['data'], omni_raw['test_data'] = datasets
    path_to_file = os.path.join(datasets_dir, 'BinaryOMNIGLOT', filename)
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    scipy.io.savemat(path_to_file, omni_raw)


def get_fixed_mnist(datasets_dir, validation_size=0):
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join(datasets_dir, 'BinaryMNIST',
                           'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(datasets_dir, 'BinaryMNIST',
                           'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(datasets_dir, 'BinaryMNIST',
                           'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')
    train_data = np.concatenate((validation_data, train_data))
    validation_data = train_data[:validation_size]
    train_data = train_data[validation_size:]

    datasets = [train_data, validation_data, test_data]
    datasets = map(shuffle, datasets)
    train_data, validation_data, test_data = map(lambda x: 255.0 * x, datasets)

    options = dict(dtype=dtypes.float32, reshape=False, seed=1234)

    train = mnist.DataSet(train_data, train_data, **options)
    validation = mnist.DataSet(validation_data, validation_data, **options)
    test = mnist.DataSet(test_data, test_data, **options)

    return base.Datasets(train=train, validation=validation, test=test)


def get_fixed_omniglot(datasets_dir, validation_size=0):
    def reshape_omni(data):
        data = data.reshape((-1, 28, 28))
        return data.reshape((-1, 28 * 28), order='fortran')
    omni_raw = scipy.io.loadmat(
        os.path.join(datasets_dir, 'chardata.mat'))
    train_data = reshape_omni(omni_raw['data'].T.astype('float32'))
    test_data = reshape_omni(omni_raw['testdata'].T.astype('float32'))
    validation_data = train_data[:validation_size]
    train_data = train_data[validation_size:]

    datasets = [train_data, validation_data, test_data]
    train_data, validation_data, test_data = map(lambda x: 255.0 * x, datasets)

    options = dict(dtype=dtypes.float32, reshape=False, seed=1234)
    train = mnist.DataSet(train_data, train_data, **options)
    validation = mnist.DataSet(validation_data, validation_data, **options)
    test = mnist.DataSet(test_data, test_data, **options)

    return base.Datasets(train=train, validation=validation, test=test)


def get_fixed_freyfaces(datasets_dir, validation_size=0, test_size=500):
    frey_raw = scipy.io.loadmat(
        os.path.join(datasets_dir, 'FreyFaces', 'frey_rawface.mat'))
    train_data = shuffle(frey_raw['ff'].T.astype('float32'))
    train_data, test_data = train_data[test_size:], train_data[:test_size]
    validation_data = train_data[:validation_size]
    train_data = train_data[validation_size:]

    datasets = [train_data, validation_data, test_data]
    train_data, validation_data, test_data = map(lambda x: x, datasets)

    options = dict(dtype=dtypes.float32, reshape=False, seed=1234)
    train = mnist.DataSet(train_data, train_data, **options)
    validation = mnist.DataSet(validation_data, validation_data, **options)
    test = mnist.DataSet(test_data, test_data, **options)

    return base.Datasets(train=train, validation=validation, test=test)


def choose_vaes_and_learning_rates(encoder_distribution, train_obj_samples,
                                   vaes_to_choose='all'):
    name_to_vae = {
        'VAE': VAE,  # noqa
        'LogDerTrickVAE': LogDerTrickVAE,  # noqa
        'NVILVAE': NVILVAE,  # noqa
        'MuPropVAE': MuPropVAE,  # noqa
        'GumbelSoftmaxTrickVAE': GumbelSoftmaxTrickVAE  # noqa
    }
    vaes = []
    if (train_obj_samples > 1) and ((vaes_to_choose == 'all')
                                    or ('VIMCOVAE' in vaes_to_choose)):
        vaes.append(VIMCOVAE)  # noqa
        if vaes_to_choose != 'all':
            vaes_to_choose.remove('VIMCOVAE')
    if encoder_distribution == 'gaussian':
        if vaes_to_choose == 'all':
            vaes += [VAE, LogDerTrickVAE, NVILVAE, MuPropVAE]  # noqa
        else:
            vaes += [name_to_vae[name] for name in vaes_to_choose]
    elif encoder_distribution == 'multinomial':
        if vaes_to_choose == 'all':
            vaes += [LogDerTrickVAE, NVILVAE, MuPropVAE, GumbelSoftmaxTrickVAE]  # noqa
        else:
            vaes += [name_to_vae[name] for name in vaes_to_choose]
    return vaes


def setup_vaes_and_params(X_train, X_val, X_test, dataset, n_z, n_ary,
                          encoder_distribution, learning_rate, nonlinearity,
                          train_batch_size, train_obj_samples, val_batch_size,
                          val_obj_samples, test_batch_size, test_obj_samples,
                          cuda_devices, save_step, n_epochs=3001,
                          save_weights=True, mem_fraction=0.333, all_vaes=True,
                          mode='train', results_dir=None,
                          logging_path='logging.txt', learning_rates=None,
                          vaes_to_choose='all', n_iterations=10, weights=None):
    n_input = X_train.images.shape[1]
    n_hidden = 200
    network_architecture = {
        'encoder': {
            'h1': (n_input, n_hidden),
            'h2': (n_hidden, n_ary * n_hidden),
            'out_mean': (n_ary * n_hidden, n_ary * n_z)
        },
        'decoder': {
            'h1': (n_ary * n_z, n_hidden),
            'h2': (n_hidden, n_hidden),
            'out_mean': (n_hidden, n_input)
        }
    }
    if encoder_distribution == 'gaussian':
        layer_sizes = (n_ary * n_hidden, n_ary * n_z)
        network_architecture['encoder']['out_log_sigma_sq'] = layer_sizes

    train_params = {
        'data': X_train,
        'n_samples': X_train.num_examples,
        'batch_size': train_batch_size,
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
        'save_path': 'weights',
        'cuda_devices': cuda_devices,
        'mem_fraction': mem_fraction,
        'results_dir': results_dir,
        'logging_path': logging_path,
        'learning_rates': learning_rates,
        'n_iterations': n_iterations,
        'weights': weights
    }
    vae_params = {
        'dataset': dataset,
        'n_input': n_input,
        'n_z': n_z,
        'n_ary': n_ary,
        'n_samples': train_params['obj_samples'],
        'encoder_distribution': encoder_distribution,
        'network_architecture': network_architecture,
        'learning_rate': learning_rate or learning_rates[0],
        'nonlinearity': nonlinearity
    }

    vaes = choose_vaes_and_learning_rates(encoder_distribution,
                                          train_obj_samples,
                                          vaes_to_choose=vaes_to_choose)

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
    elif mode == 'visualize':
        input_vaes_and_params.pop('val_params')
        input_vaes_and_params.pop('test_params')
    return input_vaes_and_params
