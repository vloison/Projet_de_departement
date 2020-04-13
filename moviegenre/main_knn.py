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
from utils.accuracy import multi_label

# Variables reloadées en attendant l'intégration à la pipeline

training_size = 2000  # Size of the training set
testing_size = 100   # Size of the testing set

MOVIES = pd.read_csv('../data/2010-1-1_2020-1-1_17.csv')
GENRES_DICT={'Action': 0,
 'Animation': 1, 
 'Aventure': 2, 
 'Biopic': 3, 
 'Comédie': 4, 
 'Comédie-dramatique': 5, 
 'Comédie-musicale': 6, 
 'Documentaire': 7,
 'Drame': 8, 
 'Epouvante-horreur': 9, 
 'Fantastique': 10, 
 'Historique': 11, 
 'Policier': 12, 
 'Romance': 13, 
 'Science-fiction': 14, 
 'Thriller': 15, 
 'Western': 16}
IMAGE_SIZE = [150, 100, 3]
str_im_size = '150-100-3'
begin_date = '2010-1-1'
end_date = '2020-1-1'
nb_genres = 17

# LOAD DATA
fileend  = '_'+'tr'+str(training_size)+'t'+str(testing_size)+'_'+str_im_size+'_'+begin_date+'_'+end_date+'_'+str(nb_genres)+'.npy'
Xtr = np.load('../data/splitted/xtr_uniform' + fileend)
print("Shape of Xtr:", Xtr.shape)
Xtest = np.load('../data/splitted/xtest_uniform' + fileend)
print("Shape of Xtest:", Xtest.shape)
Xtrprim = np.reshape(Xtr, (len(Xtr), 150*100*3))
print("Shape of Xtrprim:", Xtrprim.shape)
Ytr = np.load('../data/splitted/ytr_uniform' + fileend)
print("Shape of Ytr", Ytr.shape)
Ytest = np.load('../data/splitted/ytest_uniform' + fileend)
print("Shape of Ytest", Ytest.shape)
training_set = np.load('../data/splitted/idtr_uniform' + fileend)
testing_set = np.load('../data/splitted/idtest_uniform'+fileend)


k=3

print('accuracy', 
      test_KNN(MOVIES, Xtr, Xtrprim, Ytr, training_set, Xtest, Ytest, testing_set, k, multi_label, IMAGE_SIZE, GENRES_DICT))