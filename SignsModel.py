# File name: SignsModel
# Copyright 2017 Chidambaram Periakaruppan
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import h5py
import matplotlib.pyplot as plt
import numpy as np

from Model import Model
from NNTF import NNTF

np.random.seed(1)


class SignsModel(Model):
    '''
    Class implementation for signs model.
    '''

    def load_data(self, train_path, test_path):
        '''

        Loads the signs data given the paths.

        Args:
            train_path(str): Training data file path.
            test_path(str): Testing data file path.

        Returns:

        '''
        train_dataset = h5py.File(train_path, 'r')
        test_dataset = h5py.File(test_path, 'r')
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

        train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0], -1)
        test_set_y_orig = test_set_y_orig.reshape(test_set_y_orig.shape[0], -1)
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def transform_data(self):
        '''
        Transforms the original data so that they are normalized.
        Transforms the Y data into one hot encoding as well with samples in rows.
        Returns:
            tuple: train_x(ndarray): Transformed X training data.
            train_y(ndarray): Transformed Y training data.
            test_x(ndarray): Transformed X test data.
            test_y(ndarray): Transformed Y test data.

        '''
        m = self.train_X_orig.shape[0]
        train_x = self.train_X_orig / 255.

        m_test = self.test_X_orig.shape[0]
        test_x = self.test_X_orig / 255.

        classes = max(self.classes)+1
        train_y = np.zeros((m, classes))
        train_y[np.arange(m), self.train_Y_orig.flatten()] = 1

        test_y = np.zeros((m_test, classes))
        test_y[np.arange(m_test), self.test_Y_orig.flatten()] = 1

        return train_x, train_y,  test_x, test_y

    def show_errors(self, num_errors=5):
        '''
        Shows the errors.

        Args:
            num_errors: Number of errors to show.

        Returns:

        '''
        super(SignsModel, self).show_errors()

        show_num_errors = min(num_errors, np.sum(self.errors * 1))

        classification = self.test_Y[self.errors]
        prediction = self.test_Y_pred[self.errors]
        images = self.test_X_orig[self.errors]

        for i in range(0, show_num_errors):
            self.show_data(i, 3, images, classification)
            print('Prediction is %s' % self.classes[prediction[i]])

    def show_data(self, index, size=6, X=np.array([]), Y=np.array([])):
        '''

        Shows picture of a given index from a dataset and it's label.

        Args:
            index(int): Data sample to show.
            size(int): Size of the image.
            X: X of dataset.
            Y: Y of dataset.


        '''

        if X.shape[0] == 0:
            X = self.train_X_orig
        if Y.shape[0] == 0:
            Y = self.train_Y
        classes = self.classes

        plt.rcParams['figure.figsize'] = (size, size)
        plt.imshow(X[index, :])
        plt.show()

        classification = np.where(Y[index])
        print('This is a %s' % classification)


def unit_test():
    '''

    Runs the coursera unit test for the signs dataset.

    '''

    signs_model = SignsModel('./datasets/signs/train_signs.h5', './datasets/signs/test_signs.h5', unit_test=True)
    # signs_model.show_data(2)

    layers = [
        ['conv2d', {'f':4, 'n_f':8, 's': 1, 'padding': 'SAME'}],
        ['relu'],
        ['maxpool', {'f':8, 's': 8, 'padding': 'SAME'}],
        ['conv2d', {'f': 2, 'n_f': 16, 's': 1, 'padding': 'SAME'}],
        ['relu'],
        ['maxpool', {'f': 4, 's': 4, 'padding': 'SAME'}],
        ['flatten'],
        ['fullyconnected']
    ]
    learning_rate = 0.009
    epochs = 100
    mini_batch_size = 64
    print_cost = True
    beta1=0.9
    beta2=0.999

    signs_model.train(NNTF, layers, learning_rate, epochs, mini_batch_size, 3,
                    print_cost=print_cost, beta1=beta1, beta2=beta2)
    signs_model.predict_train()
    signs_model.predict_test()

    # signs_model.show_errors()

    expected_result = {'J': 0.18159555457532406, 'train': 0.92962962, 'test': 0.79166669}
    print('Signs model result', str(signs_model.unit_test))
    print('Signs model expected', str(expected_result))
    if str(signs_model.unit_test) == str(expected_result):
        print("Signs model unit test: OK!!!")
    else:
        print("Signs model results don't match expected results. Please check!!!")
    return


if __name__ == "__main__":
    unit_test()
