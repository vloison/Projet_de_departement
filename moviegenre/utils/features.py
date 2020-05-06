# -*- coding: utf-8 -*-
import numpy as np 

def normalize_features(train_features, test_features):
    """Normalizes features for kNN"""
    train_features_norm = np.zeros(train_features.shape)
    test_features_norm = np.zeros(test_features.shape)
    for i in range(len(train_features_norm)):
        train_features_norm[i] = train_features[i]/np.linalg.norm(train_features[i])
    for i in range(len(test_features)):
        test_features_norm[i] = test_features[i]/np.linalg.norm(test_features[i])
    return(train_features_norm, test_features_norm)



