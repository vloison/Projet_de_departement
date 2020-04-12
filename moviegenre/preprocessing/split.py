# -*- coding: utf-8 -*-
"""
Split data into training set and testing set.
"""
import numpy as np


def prepare_unif_sets(database, X, Y, IDS, GENRES_DICT, training_size, testing_size):
    print(X.shape, Y.shape, IDS.shape, len(GENRES_DICT), training_size, testing_size)
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
        print(len(IDStr), training_size, len(train_candidates))
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
    Xtr = np.zeros((len(IDStr), X.shape[1], X.shape[2], X.shape[3]))
    Ytr = np.zeros((len(IDStr), len(GENRES_DICT)))
    Xtest = np.zeros((len(IDStest), X.shape[1], X.shape[2], X.shape[3]))
    Ytest = np.zeros((len(IDStest), len(GENRES_DICT)))
    for i in range(len(IDStr)):
        Xtr[i] = X[np.argwhere(IDS == IDStr[i])]
        Ytr[i] = Y[np.argwhere(IDS == IDStr[i])]
    for i in range(len(IDStest)):
        Xtest[i] = X[np.argwhere(IDS == IDStest[i])]
        Ytest[i] = Y[np.argwhere(IDS == IDStest[i])]
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
    return Xtr, Ytr, IDStr, Xtest, Ytest, IDStest


def split_data(database, posters, genres, ids, genres_dict, training_size, testing_size, split_method, verbose=True, logger=None):
    if split_method == 'uniform': return prepare_unif_sets(database, posters, genres, ids, genres_dict, training_size, testing_size)
    return prepare_sets(posters, genres, ids, training_size, testing_size)
