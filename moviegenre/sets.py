# -*- coding: utf-8 -*-
"""
In this file, we create training and testing sets and save them.
"""
import numpy as np
import pandas as pd
from utils.preprocessing import read_csv_with_genres
from utils.constants import TRAINING_SIZE, TESTING_SIZE, GENRES_DICT, CLEAN_MOVIES_PATH, SIZE
from utils.display import show_img

# Definition of the training and testing sets


def prepare_unif_sets(database, X, Y, IDS, training_size, testing_size):
    nb_genres = len(GENRES_DICT)
    IDStr = []
    IDStest = []
    restants = list(GENRES_DICT)
    # On commence par créer les listes des indices des éléments des sets

    # Première boucle pour repérer les genres de cardinal non suffisant.
    # Pour chaque genre non suffisant, on met tous les films de ce genre dans
    # le training_set, sauf 1 qu'on met dans le testing_set
    for k in restants:
        # Récupérer tous les films du genre concerné
        tab_genre = database[database['genres'].str[0] == k]
        card_genre = tab_genre.shape[0]
        # Traitement du cas où le genre a un card non suffisant
        if card_genre < training_size/nb_genres:
            # Shuffle avant extraction
            tab_genre = tab_genre.sample(n=card_genre)
            IDStr += list(tab_genre[:card_genre-1].index)
            IDStest += list(tab_genre[card_genre-1:].index)
            restants.remove(k)
    # Deuxième boucle pour les genres de cardinal suffisant.
    # Si un genre n'a pas un cardinal suffisant par rapport à la nouvelle
    # taille, on lui applique le même traitement qu'à la boucle précédente.
    new_size = int((training_size - len(IDStr))/len(restants)) + 1
    for k in restants:
        print(k)
        tab_genre = database[database['genres'].str[0] == k]
        card_genre = tab_genre.shape[0]
        if tab_genre.shape[0] < new_size:
            tab_genre = tab_genre.sample(n=card_genre)
            IDStr += list(tab_genre[:card_genre-1].index)
            IDStest += list(tab_genre[card_genre-1:].index)
            restants.remove(k)
        else:
            tab_genre = tab_genre.sample(n=new_size)
            IDStr += list(tab_genre.index)
    new_tr_size = len(IDStr)
    
    # On complète le training_set pour qu'il ait la bonne taille:
    if len(IDStr) < training_size:
        train_candidates = database.drop(database[database.index.isin(
                IDStr + IDStest)].index)
        IDStr += list(train_candidates.sample(n=training_size-len(IDStr)).index)    
    
    # On complète le testing_set pour qu'il ait la bonne taille
    test_candidates = database.drop(database[database.index.isin(
            IDStr + IDStest)].index)
    IDStest += list(test_candidates.sample(
            n=testing_size-len(IDStest)).index)
    IDStr = np.array(IDStr)
    np.random.shuffle(IDStr)
    IDStest = np.array(IDStest)
    np.random.shuffle(IDStest)
    print('taille de IDStr', IDStr.shape)
    print('taille de IDStest', IDStest.shape)

    # On construit les listes de données et labels à partir des listes d'IDS
    Xtr = np.zeros((len(IDStr), SIZE[0], SIZE[1], SIZE[2]))
    Ytr = np.zeros((len(IDStr), len(GENRES_DICT)))
    Xtest = np.zeros((len(IDStest), SIZE[0], SIZE[1], SIZE[2]))
    Ytest = np.zeros((len(IDStest), len(GENRES_DICT)))
    for i in range(len(IDStr)):
        Xtr[i] = X[np.argwhere(IDS == IDStr[i])]
        Ytr[i] = Y[np.argwhere(IDS == IDStr[i])]
    for i in range(len(IDStest)):
        Xtest[i] = X[np.argwhere(IDS == IDStest[i])]
        Ytest[i] = Y[np.argwhere(IDS == IDStest[i])]

    np.save(
        '../data/sets/unif_Xtr_tr={}_test={}.npy'.format(training_size, testing_size),
        Xtr
    )
    np.save(
        '../data/sets/unif_Ytr_tr={}_test={}.npy'.format(training_size, testing_size),
        Ytr
            )
    np.save(
        '../data/sets/unif_IDStr_tr={}_test={}.npy'.format(training_size, testing_size),
        IDStr
    )
    np.save(
        '../data/sets/unif_Xtest_tr={}_test={}.npy'.format(training_size, testing_size),
        Xtest
    )
    np.save(
        '../data/sets/unif_Ytest_tr={}_test={}.npy'.format(training_size, testing_size),
        Ytest
    )
    np.save(
        '../data/sets/unif_IDStest_tr={}_test={}.npy'.format(training_size, testing_size),
        IDStest
    )
    return Xtr, Ytr, IDStr, Xtest, Ytest, IDStest


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
    MOVIES = read_csv_with_genres(CLEAN_MOVIES_PATH)
    # Load data
    X = np.load('../data/numpy_posters.npy')
    print("Shape of X:", X.shape)
    Y = np.load('../data/numpy_genres.npy')
    print("Shape of Y", Y.shape)
    IDS = np.load('../data/numpy_ids.npy')
    Xtr, Ytr, IDStr, Xtest, Ytest, IDStest = prepare_unif_sets(MOVIES, X, Y, IDS, TRAINING_SIZE, TESTING_SIZE)
    print('Shape of Xtr', Xtr.shape)
    print('Shape of Ytr', Ytr.shape)
    show_img(MOVIES, Xtr, Ytr, IDStr, 0)
    print(Ytr[0])
    