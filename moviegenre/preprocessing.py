# -*- coding: utf-8 -*-
"""Preprocesses the input data"""
from utils.preprocessing import read_csv_with_genres, get_id, normalize
from utils.constants import GENRES_DICT, CLEAN_MOVIES_PATH, SAVELOCATION, SIZE

from pathlib import Path
import imageio
import skimage.transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def prepare_data(dir_path, dataset, size=(150, 100, 3), save=True):
    """Generates the data to be used by the neural network"""
    print('Generating dataset...')

    nb_genres = len(GENRES_DICT)
    image_glob = Path(dir_path).glob("*.jpg")
    posters, genres, ids = [], [], []
    for path in tqdm(sorted(image_glob)):
        try:
            # Meilleure gestion d'erreur à faire
            posters.append(normalize(imageio.imread(path), size))
            index = get_id(path)
            vect_genre = np.zeros(nb_genres, dtype=int)
            # Rajoute un 1 à l'indice correspondant à la position
            # du premier genre de ce film dans la liste des genres
            for genre_name in dataset.at[index, 'genres']:
                vect_genre[GENRES_DICT[genre_name]] = 1
            genres.append(vect_genre)
            ids.append(index)
        except Exception as e:
            print("Erreur", e)
    if save:
        np.save('../data/numpy_posters.npy', posters)
        np.save('../data/numpy_genres.npy', genres)
        np.save('../data/numpy_ids.npy', ids)
    print('Done.')
    return posters, genres, ids


if __name__ == "__main__":
    MOVIES = read_csv_with_genres(CLEAN_MOVIES_PATH)
    #X, Y, IDS = prepare_data(SAVELOCATION, MOVIES, SIZE)
    X, Y, IDS = np.load('../data/numpy_posters.npy'), np.load('../data/numpy_genres.npy'), np.load('../data/numpy_ids.npy')
    print(X.shape, Y.shape, IDS.shape)