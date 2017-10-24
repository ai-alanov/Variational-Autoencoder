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
from utils import binarized_mnist_fixed_binarization, setup_input_vaes_and_params

def main():
    parser = OptionParser(usage="usage: %prog [options]",
                          version="%prog 1.0")
    parser.add_option("--n_z", dest="n_z", type='int', help="size of stochastic layer")
    parser.add_option("--n_ary", dest="n_ary", type='int', help="arity of latent variable")
    parser.add_option("--en_dist", dest="encoder_distribution", type='string', help="encoder distribution")
    parser.add_option("--l_r", dest="learning_rate", type='float', help="learning rate")
    parser.add_option("--n_samples", dest="train_obj_samples", type='int', help="train objective samples")
    parser.add_option("--c_devs", dest="cuda_devices", type='string', help="cuda devices")
    (options, args) = parser.parse_args()
    
    binarized_mnist = binarized_mnist_fixed_binarization('datasets/', validation_size=10000)
    X_train, X_val, X_test = binarized_mnist.train, binarized_mnist.validation, binarized_mnist.test

    params = {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val, 
        'dataset': 'BinaryMNIST',
        'n_z': options.n_z,
        'n_ary': options.n_ary,
        'encoder_distribution': options.encoder_distribution,
        'learning_rate': options.learning_rate,
        'train_batch_size': 128,
        'train_obj_samples': options.train_obj_samples,
        'val_batch_size': 1024,
        'val_obj_samples': 5, 
        'test_batch_size': 1024,
        'test_obj_samples': 5,        
        'cuda_devices': options.cuda_devices,
        'save_step': 100,
        'n_epochs': 3001,
        'save_weights': True,
        'mem_fraction': 0.5,
        'all_vaes': True, 
        'mode': 'train', 
        'results_dir': 'test_results'
    }

    train_model(**setup_input_vaes_and_params(**params))

if __name__ == '__main__':
    main()