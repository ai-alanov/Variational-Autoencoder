#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from datetime import datetime
from optparse import OptionParser
import sys
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa

sys.path.append('../python/')
from utils import makedirs  # noqa
from train_process import train_model, test_model, plot_stds  # noqa
from train_process import get_fixed_mnist, get_fixed_omniglot  # noqa
from train_process import setup_vaes_and_params  # noqa

np.random.seed(1234)
tf.set_random_seed(1234)
sns.set_style('whitegrid')
sns.set_context(
    'paper',
    font_scale=2.0,
    rc={'lines.linewidth': 2, 'lines.markersize': 10, 'figsize': (5, 4.8)})

tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.DEBUG)
tf.logging.set_verbosity(tf.logging.FATAL)

warnings.filterwarnings('ignore', module='matplotlib')


def lrs_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, list(map(float, value.split(','))))


def split_by_comma_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


def nonlinearity_callback(option, opt, value, parser):
    if value == 'softplus':
        setattr(parser.values, option.dest, tf.nn.softplus)
    elif value == 'tanh':
        setattr(parser.values, option.dest, tf.nn.tanh)
    else:
        raise ValueError('Invalid nonlinearity: {}'.format(value))


def stage_to_epochs_callback(option, opt, value, parser):
    func, parameter = value.split(',')
    parameter = int(parameter)
    if func == 'lin':
        setattr(parser.values, option.dest, lambda x: parameter * x)
    elif func == 'exp':
        setattr(parser.values, option.dest, lambda x: parameter ** x)
    else:
        error_message = 'Invalid function type (only \'lin\', \'exp\'): {}'
        raise ValueError(error_message.format(func))


def lr_decay_callback(option, opt, value, parser):
    func, *parameters = value.split(',')
    parameters = list(map(lambda x: int(x), parameters))
    if func == 'lin':
        setattr(parser.values, option.dest, lambda x: x / parameters[0])
    elif func == 'exp':
        setattr(parser.values, option.dest, lambda x: parameters[0] ** (-x))
    elif func == 'linexp':
        setattr(parser.values, option.dest,
                lambda x: parameters[0] ** (-x / parameters[1]))
    else:
        error_message = 'Invalid function type '
        error_message += '(only \'lin\', \'exp\', \'linexp\'): {}'
        raise ValueError(error_message.format(func))


def main():
    parser = OptionParser(usage="usage: %prog [options]",
                          version="%prog 1.0")
    parser.add_option("--n_z", type='int', help="size of stochastic layer")
    parser.add_option("--n_ary", type='int', help="arity of latent variable")
    parser.add_option("--en_dist", type='string', help="encoder distribution")
    parser.add_option("--l_r", type='float', help="learning rate")
    parser.add_option("--l_rs", type='string', default=[0.001],
                      help="learning rates for validation",
                      action='callback', callback=lrs_callback)
    parser.add_option("--n_samples", type='int',
                      help="train objective samples")
    parser.add_option("--train_b_size", type='int', default=128,
                      help="train batch size")
    parser.add_option("--test_b_size", type='int', default=1024,
                      help="test batch size")
    parser.add_option("--test_n_samples", type='int', default=5,
                      help="test objective samples")
    parser.add_option("--c_devs", type='string', help="cuda devices")
    parser.add_option("--mem_frac", type='float', default=0.5,
                      help="memory fraction used in gpu")
    parser.add_option("--dataset", type='string', default='mnist',
                      help='dataset name')
    parser.add_option("--mode", type='string', default='train',
                      help='train or test')
    parser.add_option("--vaes", type='string', default='all',
                      help="vaes of training or testing",
                      action='callback', callback=split_by_comma_callback)
    parser.add_option("--n_iters", type='int', default=10,
                      help="number of iterations for gradient std estimation")
    parser.add_option("--save_step", type='int', default=100,
                      help="step of saving weights")
    parser.add_option("--display_step", type='int', default=100,
                      help="step of logging results")
    parser.add_option("--nonlinearity", type='string', default=tf.nn.softplus,
                      help="nonlinearity for VAE", action='callback',
                      callback=nonlinearity_callback)
    parser.add_option("--weights", type='string', default=None,
                      help="weight names for plot stds", action='callback',
                      callback=split_by_comma_callback)
    parser.add_option("--n_epochs", type='int', default=3001,
                      help="number of training epochs")
    parser.add_option("--n_stages", type='int', default=None,
                      help="number of training stages")
    parser.add_option("--stage_to_epochs", type='string', default=lambda x: 1,
                      help='''function which mapes a stage to the number of
                      epochs, options: lin,a -> a*stage; exp,a -> a**stage''',
                      action='callback', callback=stage_to_epochs_callback)
    parser.add_option("--valid_size", type='int', default=1345,
                      help="validation size")
    parser.add_option("--lr_decay", type='string', default=lambda x: 1,
                      help='''function which mapes a stage to the learning rate
                      decay, options: lin,a -> stage/a; exp,a -> a**(-stage),
                      linexp,a,b -> a**(-stage/b)''',
                      action='callback', callback=lr_decay_callback)
    (options, args) = parser.parse_args()
    options_to_str = ['{}:{}'.format(k[:3], v)
                      for k, v in vars(options).items() if not callable(v)]
    logging_file = '-'.join(sorted(options_to_str))
    logging_file += datetime.now().strftime("_%H:%M:%S") + '.txt'
    log_dir = os.path.join('logs', datetime.now().strftime("%Y-%m-%d"))
    makedirs(log_dir)
    logging_path = os.path.join(log_dir, logging_file)

    if options.dataset == 'BinaryMNIST':
        data = get_fixed_mnist('datasets/', validation_size=10000)
    elif 'OMNIGLOT' in options.dataset:
        data = get_fixed_omniglot(os.path.join('datasets', options.dataset),
                                  validation_size=options.valid_size)
    X_train, X_val, X_test = data.train, data.validation, data.test

    params = {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'dataset': options.dataset,
        'n_z': options.n_z,
        'n_ary': options.n_ary,
        'encoder_distribution': options.en_dist,
        'learning_rate': options.l_r,
        'nonlinearity': options.nonlinearity,
        'train_batch_size': options.train_b_size,
        'train_obj_samples': options.n_samples,
        'val_batch_size': 1024,
        'val_obj_samples': 5,
        'test_batch_size': options.test_b_size,
        'test_obj_samples': options.test_n_samples,
        'cuda_devices': options.c_devs,
        'save_step': options.save_step,
        'display_step': options.display_step,
        'n_stages': options.n_stages if options.n_stages else 1,
        'stage_to_epochs': (options.stage_to_epochs if options.n_stages
                            else lambda x: options.n_epochs),
        'save_weights': True,
        'mem_fraction': options.mem_frac,
        'mode': options.mode,
        'results_dir': 'test_results',
        'logging_path': logging_path,
        'learning_rates': options.l_rs,
        'vaes_to_choose': options.vaes,
        'n_iterations': options.n_iters,
        'weights': options.weights,
        'lr_decay': options.lr_decay
    }
    if options.mode == 'train':
        train_model(**setup_vaes_and_params(**params))
    elif options.mode == 'test':
        test_model(**setup_vaes_and_params(**params))
    elif options.mode == 'visualize':
        plot_stds(**setup_vaes_and_params(**params))


if __name__ == '__main__':
    main()
