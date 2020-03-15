# -*- coding: utf-8 -*-
"""
In this file, we create training and testing sets and save them.
"""
import numpy as np
from preprocessing import show_img, preprocess, MOVIES

training_size = 2000  # Size of the training set
testing_size = 100   # Size of the testing set


# LOAD DATA

X = np.load('../data/numpy_posters.npy')
print("Shape of X:", X.shape)
Y = np.load('../data/numpy_genres.npy')
print("Shape of Y" + str(np.shape(Y)))
IDS = np.load('../data/numpy_ids.npy')
print("Size of IDS:", np.max(IDS))
print("IDS", IDS)


# DEFINITION OF THE TRAINING AND TESTING SETS
def sets(dataset, training_size, testing_size, IDS):
    """ Builds a training_set and a testing_set as np.arrays, lists of
    indices. Each set has the size given in argument """
    # Copy IDS to keep the order in the original
    IDS2 = [IDS[i] for i in range(len(IDS))]
    np.random.shuffle(IDS2)
    training_set = IDS2[:training_size]
    testing_set = IDS2[training_size:training_size+testing_size]
    return training_set, testing_set


def prepare_training_set(X, Y, dataset, training_size, testing_size, IDS):
    training_set, testing_set = sets(dataset, training_size, testing_size, IDS)
    Xtr = np.array([X[np.where(IDS == i)] for i in training_set])
    Xtr = np.reshape(Xtr, (len(Xtr), 150, 100, 3))
    
    Ytr = np.array([Y[np.where(IDS == i)] for i in training_set])
    Ytr = np.reshape(Ytr, (len(Ytr), 33))

    Xtest = np.array([X[np.where(IDS == i)] for i in testing_set])
    Xtest = np.reshape(Xtest, (len(Xtest), 150, 100, 3))

    Ytest = np.array([Y[np.where(IDS == i)] for i in testing_set])
    Ytest = np.reshape(Ytest, (len(Ytest), 33))

    # Save data
    np.save(
        '../data/sets/Xtr_tr='+str(training_size)+'_test='+str(testing_size)+'.npy',
        Xtr
    )
    np.save(
        '../data/sets/Ytr_tr='+str(training_size)+'_test='+str(testing_size)+'.npy',
        Ytr
            )

    np.save(
        '../data/sets/Xtest_tr=' + str(training_size) + '_test=' + str(testing_size) + '.npy',
        Xtest
    )
    np.save(
        '../data/sets/Ytest_tr=' + str(training_size) + '_test=' + str(testing_size) + '.npy',
        Ytest
    )

    #return Xtr, Ytr, training_set, testing_set

# Uncomment the following lines and run the file to create and save other 
# training_sets and testing_sets
"""Xtr, Ytr, tr, test = """
prepare_training_set(X, Y, MOVIES, training_size, testing_size, IDS)
#print('Shape of Xtr', Xtr.shape)
#print('Shape of Ytr', Ytr.shape)
