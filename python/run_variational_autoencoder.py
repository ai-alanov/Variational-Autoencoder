import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from variational_autoencoder import VariationalAutoencoder

def train(data, n_samples, n_input, n_z, 
          batch_size, learning_rate, network_architecture, 
         training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(n_input, n_z, 
                                 network_architecture, learning_rate)
    for epoch in xrange(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in xrange(total_batch):
            batch_xs, _ = data.train.next_batch(batch_size)
            cost = vae.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
            
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost =", "{:.9f}".format(avg_cost)
    return vae

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    n_samples = mnist.train.num_examples
    n_input = 784
    n_z = 20
    batch_size = 100
    learning_rate = 0.001

    network_architecture = {
        'n_hidden_encoder_1': 500,
        'n_hidden_encoder_2': 500,
        'n_hidden_decoder_1': 500,
        'n_hidden_decoder_2': 500
    }
    vae = train(mnist, n_samples, n_input, n_z, 
            batch_size, learning_rate, 
            network_architecture, training_epochs=75)