# File name: NNTF
# Copyright 2017 Chidambaram Periakaruppan
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops

class NNTF(object):


    '''Creates the TensorFlow neural network.

     Args:
         layers(list): List of layers and their paramters. eg.
            [['conv2d', {'f': 2, 'n_f': 16, 's': 1, 'padding': 'SAME'}],
            ['relu'],
            ['maxpool', {'f': 4, 's': 4, 'padding': 'SAME'}],
            ['flatten'],
            ['fullyconnected']]
            where f is the filter size, n_f is the number of filters, s is stride and padding.


     Attributes
         layers(list): List of nodes in each of the layers including the input and output layer.
         X(tf.placeholder): Placeholder for X.
         Y(tf.placeholder): Placeholder for Y.
         output_layer(None): For the final activation of the neural network.
         predcitions(None): For the final predictions.
         saver_path(str): Path to save the variables after training.


     '''

    def __init__(self, layers, X_shape, Y_shape):

        tf.reset_default_graph()
        self.layers = layers

        X_shape = list(X_shape)
        X_shape[0] = None
        Y_shape = list(Y_shape)
        Y_shape[0] = None
        self.X = tf.placeholder(shape=X_shape, dtype=tf.float32)
        self.Y = tf.placeholder(shape=Y_shape, dtype=tf.float32)

        self.output_layer = None
        self.predictions = None

        self.saver_path = "/tmp/model.ckpt"


    def initialize_conv2d(self, name, params, n_c):
        '''
        Initializes the conv2d filters with xavier initialization.

        Args:
            name(str): Name of the filter.
            params(dict): Parameters for the conv2d layer.
            n_c(int): Number of channels from the previous layer.

        Returns:
            weights(tf.variable): Iniatilzed weights.

        '''


        f = params.get('f')
        n_f = params.get('n_f')
        tf.set_random_seed(1)
        initializer = tf.contrib.layers.xavier_initializer(seed=0)
        weights = tf.get_variable('W'+name, shape=[f, f, n_c, n_f], dtype=tf.float32, initializer=initializer)

        return weights


    def nn(self):
        '''
        Create the neural network up to the ZL which will be before the final activation.
        '''

        tf_layers = {}
        tf_layers['0'] = self.X
        filters = {}
        filters['0'] = None

        Y_shape = self.Y.get_shape().as_list()
        num_outputs = Y_shape[-1]


        for i, l in enumerate(self.layers):
            prev_layer = str(i)
            curr_layer = str(i + 1)

            n_c_prev = tf_layers[prev_layer].get_shape()
            n_c_prev = n_c_prev.as_list()[-1]

            if l[0] == 'conv2d':
                s = l[1].get('s')
                padding = l[1].get('padding')
                strides = [1,s,s,1]

                filters[curr_layer] = self.initialize_conv2d(curr_layer, l[1], n_c_prev)

                tf_layers[curr_layer] = tf.nn.conv2d(input=tf_layers[prev_layer], filter=filters[curr_layer], strides=strides,padding= padding)
            elif l[0] == 'relu':
                tf_layers[curr_layer] = tf.nn.relu(features=tf_layers[prev_layer])
            elif l[0] == 'maxpool':
                f = l[1].get('f')
                s = l[1].get('s')
                padding = l[1].get('padding')
                filter = [1,f,f,1]
                strides = [1,s,s,1]

                tf_layers[curr_layer] = tf.nn.max_pool(value=tf_layers[prev_layer], ksize=filter, strides = strides, padding=padding)
            elif l[0] == 'flatten':
                tf_layers[curr_layer] = tf.contrib.layers.flatten(tf_layers[prev_layer])
            elif l[0] == 'fullyconnected':
                tf_layers[curr_layer] = tf.contrib.layers.fully_connected(tf_layers[prev_layer], num_outputs,
                                                                          activation_fn=None)
            else:
                print('no layer called %s' % l[0])


        self.output_layer = tf_layers[curr_layer]
        self.predictions = tf.argmax(self.output_layer,1)
        correct_prediction = tf.equal(self.predictions, tf.argmax(self.Y,1))
        self.accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def fit(self, train_X, train_Y, learning_rate, num_epochs=1000, mini_batch_size=32,
            beta1=0, beta2=0, seed=3, print_cost=False):
        '''
        Trains the model given a X and Y and learning paramaters and returns the final cross entropy loss.

        Args:
            train_X (ndarry): Samples as rows, features in columns.
            train_Y(ndarry): Labels in rows.
            learning_rate(float): Learning rate.
            num_epochs(int): Number of epochs.
            mini_batch_size(int): Mini-batch size.
            beta1(float): Momentum beta1, if 0 then there is no momentum.
            beta2(float): RMSprop beta2, 0 if 0 then there is no rmsprop.
            seed(int): Ramdom number generator seed.
            print_cost(boolean): True to print cost as you train.

        Returns:
            J (float): Cross Entroy loss for the given X, Y.
        '''
        # ops.reset_default_graph()
        tf.set_random_seed(1)
        self.nn()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.output_layer))

        J = cost

        # self.predictions = tf.cast(tf.greater(self.output_layer, tf.constant(0., dtype=tf.float64)), dtype=tf.int64)
        # Y_int = tf.cast(self.Y, dtype=tf.int64)
        # self.accuracy = tf.contrib.metrics.accuracy(labels=Y_int, predictions=self.predictions)



        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(J)

        init_gl = tf.global_variables_initializer()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_gl)

            for i in range(0, num_epochs):
                seed = seed + 1
                mini_batches = self.random_mini_batches(train_X, train_Y, mini_batch_size, seed)
                epoch_cost = 0.
                num_minibatches = int(train_X.shape[0]/ mini_batch_size)

                for X, Y in mini_batches:

                    _, batch_cost = sess.run([optimizer, J], feed_dict={self.X: X, self.Y: Y})

                    epoch_cost += batch_cost/num_minibatches


                if print_cost and i % 5 == 0:
                    print('Cost after epoch %s: %.6f'%(i, epoch_cost))


            saver.save(sess, self.saver_path)
            sess.close()
        return epoch_cost

    def predict(self, X, Y=np.array([])):
        '''
        Returns the predictions of the model and the accuracy score.

        Args:
            X (ndarry): Samples as columns, features in rows.
            Y(ndarry): Labels. If empty array there will be no accuracy score.


        Returns:
            tuple: Y_pred(ndarray): Predicted labels.
            accuracy(float): Accuracy score.
        '''

        saver = tf.train.Saver()

        sess = tf.Session()

        saver.restore(sess, self.saver_path)
        if Y.shape[0] != 0:

            Y_pred, accuracy = sess.run([self.predictions, self.accuracy],
                                        feed_dict={self.X: X, self.Y: Y})
            print('Accuracy is %s%%' % (accuracy * 100))

        else:
            Y_pred = sess.run(self.predictions,
                              feed_dict={self.X: X})
            accuracy = None
        return Y_pred, accuracy


    def random_mini_batches(self, X, Y, mini_batch_size, seed):
        '''
        Takes X, Y and splits into mini-batches.

        Args:
            X (ndarry): Samples as rows, features in columns.
            Y(ndarry): Labels in rows.
            mini_batch_size: Mini-batch size.
            seed(int): Ramdom number generator seed.

        Returns:
            mini_batches(list): List of pairs of X,Y mini-batches.
        '''
        if mini_batch_size:
            np.random.seed(seed)

            m = X.shape[0]

            indices = list(np.random.permutation(m))

            num_splits = m // int(mini_batch_size)
            mini_batches = []
            if num_splits > 0:
                for i in range(0, num_splits):
                    split = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
                    X_batch = X[split, :, :, :]
                    Y_batch = Y[split, :]
                    mini_batches.append((X_batch, Y_batch))

            split = indices[num_splits * mini_batch_size:]
            X_batch = X[split, :, :, :]
            Y_batch = Y[split, :]
            mini_batches.append((X_batch, Y_batch))
            return mini_batches
        else:
            return [(X, Y)]