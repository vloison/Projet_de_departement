# -*- coding: utf-8 -*-
"""
In this file, we create training and testing sets and save them.
"""
import numpy as np
from preprocessing import show_img, preprocess, MOVIES

training_size = 1000  # Size of the training set
testing_size = 100   # Size of the testing set


# LOAD DATA

X = np.load('../data/all_clean_posters.npy')
print("Shape of X:", X.shape)
Y = np.load('../data/all_clean_genres.npy')
print("Shape of Y" + str(np.shape(Y)))
IDS = np.load('../data/all_clean_ids.npy')
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
    # Save data
    np.save('./sets/Xtr_tr='+str(training_size)+'_test='+str(testing_size)+'.npy'
            , Xtr)
    np.save('./sets/Ytr_tr='+str(training_size)+'_test='+str(testing_size)+'.npy'
            , Ytr)
    np.save('./sets/training_set_tr='+str(training_size)+'_test='+str(testing_size)+'.npy'
            , training_set)
    np.save('./sets/testing_set_tr='+str(training_size)+'_test='+str(testing_size)+'.npy'
            , testing_set)
    return Xtr, Ytr, training_set, testing_set

# Uncomment the following lines and run the file to create and save other 
# training_sets and testing_sets
Xtr, Ytr, tr, test = prepare_training_set(X, Y, MOVIES, training_size, testing_size, IDS)
#print('Shape of Xtr', Xtr.shape)
#print('Shape of Ytr', Ytr.shape)
