# -*- coding: utf-8 -*-
# RENOMMER LE FICHIER!!!
"""Preprocesses the input data"""
from pathlib import Path
from skimage import transform
import imageio
import numpy as np
from tqdm import tqdm


def get_id(path):
    """Gets the id from the Pathlib path"""
    filename = path.parts[-1]
    index_f = filename.rfind(".jpg")
    return int(filename[:index_f])


def normalize(img, size):
    """Normalizes the image"""
    img = transform.resize(img, size)
    img = img.astype(np.float32)
#     img = (img / 127.5) -1
    return img


def preprocess_data(dir_path, dataset, genres_dict, size, verbose=True, logger=None):
    """Generates the data to be used by the neural network"""
    nb_genres = len(genres_dict)
    generator = dataset.iterrows()
    if verbose:
        print('Generating dataset...')
        generator = tqdm(generator, total=len(dataset))
    posters, genres, ids = [], [], []
    for index, row in generator:
        path = dir_path/(str(index)+'.jpg')
        try:
            # MEILLEURE GESTION D'ERREUR À FAIRE
            posters.append(normalize(imageio.imread(path), size))
            vect_genre = np.zeros(nb_genres, dtype=int)
            for genre_name in dataset.at[index, 'genres']:
                vect_genre[genres_dict[genre_name]] = 1
            genres.append(vect_genre)
            ids.append(index)
        except Exception as e:
            if verbose:
                print("Erreur {} with film {}".format(e, index))
    return np.array(posters), np.array(genres), np.array(ids)
