import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1234)
tf.set_random_seed(1234)
sns.set_style('whitegrid')
sns.set_context(
    'paper',
    font_scale=2.0,
    rc={'lines.linewidth': 2, 'lines.markersize': 10, 'figsize': (5, 4.8)})

from tensorflow.examples.tutorials.mnist import input_data 
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from optparse import OptionParser
import sys
sys.path.append('../python/')
import os

tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.DEBUG)
tf.logging.set_verbosity(tf.logging.FATAL)

import warnings
warnings.filterwarnings('ignore', module='matplotlib')

from vae import VAE, LogDerTrickVAE, VIMCOVAE, NVILVAE, MuPropVAE, GumbelSoftmaxTrickVAE
from utils import train_model, test_model, consider_stds, get_gradient_mean_and_std
from utils import binarized_mnist_fixed_binarization, setup_input_vaes_and_params, makedirs

def main():
    parser = OptionParser(usage="usage: %prog [options]",
                          version="%prog 1.0")
    parser.add_option("--n_z", type='int', help="size of stochastic layer")
    parser.add_option("--n_ary", type='int', help="arity of latent variable")
    parser.add_option("--en_dist", type='string', help="encoder distribution")
    parser.add_option("--l_r", type='float', help="learning rate")
    parser.add_option("--n_samples", type='int', help="train objective samples")
    parser.add_option("--c_devs", type='string', help="cuda devices")
    parser.add_option("--mem_frac", type='float', default=0.5, help="memory fraction used in gpu")
    (options, args) = parser.parse_args()
    logging_file = '-'.join(sorted(['{}{}'.format(k, v) for k, v in vars(options).items()]))
    logging_file += datetime.now().strftime("_%H:%M:%S") + '.txt'
    log_dir = os.path.join('logs', datetime.now().strftime("%Y-%m-%d"))
    makedirs(log_dir)
    logging_path = os.path.join(log_dir, logging_file)
    
    binarized_mnist = binarized_mnist_fixed_binarization('datasets/', validation_size=10000)
    X_train, X_val, X_test = binarized_mnist.train, binarized_mnist.validation, binarized_mnist.test

    params = {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val, 
        'dataset': 'BinaryMNIST',
        'n_z': options.n_z,
        'n_ary': options.n_ary,
        'encoder_distribution': options.en_dist,
        'learning_rate': options.l_r,
        'train_batch_size': 128,
        'train_obj_samples': options.n_samples,
        'val_batch_size': 1024,
        'val_obj_samples': 5, 
        'test_batch_size': 1024,
        'test_obj_samples': 5,        
        'cuda_devices': options.c_devs,
        'save_step': 100,
        'n_epochs': 3001,
        'save_weights': True,
        'mem_fraction': options.mem_frac,
        'all_vaes': True, 
        'mode': 'train', 
        'results_dir': 'test_results',
        'logging_path': logging_path
    }

    train_model(**setup_input_vaes_and_params(**params))

if __name__ == '__main__':
    main()