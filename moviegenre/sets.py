# -*- coding: utf-8 -*-
"""
In this file, we create training and testing sets and save them.
"""
import numpy as np


# Definition of the training and testing sets

def prepare_sets(X, Y, IDS, training_size, testing_size):
    """ Builds a training_set and a testing_set as np.arrays, lists of
    indices. Each set has the size given in argument """
    permutation = np.random.permutation(training_size+testing_size)
    IDStr = IDS[permutation[:training_size]]
    IDStest = IDS[permutation[training_size:training_size+testing_size]]
    Xtr = X[permutation[:training_size]]
    Xtest = X[permutation[training_size:training_size+testing_size]]
    Ytr = Y[permutation[:training_size]]
    Ytest = Y[permutation[training_size:training_size+testing_size]]

    np.save(
        '../data/sets/Xtr_tr={}_test={}.npy'.format(training_size, testing_size),
        Xtr
    )
    np.save(
        '../data/sets/Ytr_tr={}_test={}.npy'.format(training_size, testing_size),
        Ytr
            )
    np.save(
        '../data/sets/IDStr_tr={}_test={}.npy'.format(training_size, testing_size),
        IDStr
    )
    np.save(
        '../data/sets/Xtest_tr={}_test={}.npy'.format(training_size, testing_size),
        Xtest
    )
    np.save(
        '../data/sets/Ytest_tr={}_test={}.npy'.format(training_size, testing_size),
        Ytest
    )
    np.save(
        '../data/sets/IDStest_tr={}_test={}.npy'.format(training_size, testing_size),
        IDStest
    )
    return Xtr, Ytr, IDStr, Xtest, Ytest, IDStest


if __name__ == '__main__':
    # Load data
    X = np.load('../data/numpy_posters.npy')
    print("Shape of X:", X.shape)
    Y = np.load('../data/numpy_genres.npy')
    print("Shape of Y", Y.shape)
    IDS = np.load('../data/numpy_ids.npy')
    Xtr, Ytr, IDStr, Xtest, Ytest, IDStest = prepare_sets(X, Y, IDS, TRAINING_SIZE, TESTING_SIZE)
    print('Shape of Xtr', Xtr.shape)
    print('Shape of Ytr', Ytr.shape)
