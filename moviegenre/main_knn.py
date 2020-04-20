# -*- coding: utf-8 -*-
"""
kNN
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from knn.knn import KNN, test_KNN
from utils.accuracy import mono_label_KNN

# Variables reloadées en attendant l'intégration à la pipeline

MOVIES = pd.read_csv('../data/clean_poster_data_7.csv')
GENRES_DICT = {'Action': 0,
 'Animation': 1, 
 'Comédie': 2, 
 'Comédie-dramatique': 3,  
 'Documentaire': 4,
 'Drame': 5,      
 'Thriller': 6}
IMAGE_SIZE = [100, 100, 3]

nb_genres = 7

# LOAD DATA
fileend = '_s700t0.15_100-100-3_7.npy'
XTR = np.load('../data/sets/xtr' + fileend)
print("Shape of XTR:", XTR.shape)
XTEST = np.load('../data/sets/xtest' + fileend)
print("Shape of XTEST:", XTEST.shape)

TRAINING_FEATURES = np.reshape(XTR, (len(XTR), IMAGE_SIZE[0]*IMAGE_SIZE[1]*IMAGE_SIZE[2]))
print("Shape of training features:", TRAINING_FEATURES.shape)
TESTING_FEATURES = np.reshape(XTEST, (len(XTEST), IMAGE_SIZE[0]*IMAGE_SIZE[1]*IMAGE_SIZE[2]))
print("Shape of testing features:", TESTING_FEATURES.shape)

YTR = np.load('../data/sets/ytr' + fileend)
print("Shape of YTR", YTR.shape)
YTEST = np.load('../data/sets/ytest' + fileend)
print("Shape of YTEST", YTEST.shape)

TRAINING_IDS = np.load('../data/sets/idtr' + fileend)
TESTING_IDS = np.load('../data/sets/idtest'+fileend)

k = 5
ind = 26
#predictions_test = KNN(MOVIES, XTR, TRAINING_FEATURES, YTR, TRAINING_IDS, XTEST, TESTING_FEATURES, TESTING_IDS, -1, k, IMAGE_SIZE, print_results=True)

#print('accuracy', 
#         test_KNN(MOVIES, XTR, TRAINING_FEATURES, YTR, TRAINING_IDS, XTEST, TESTING_FEATURES, YTEST, TESTING_IDS, k, mono_label_KNN, IMAGE_SIZE, GENRES_DICT))