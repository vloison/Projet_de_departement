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


def preprocess_data(dir_path, dataset, nb_genres, size, verbose=True, logger=None):
    """Generates the data to be used by the neural network"""
    if verbose:
        print('Generating dataset...')
    image_glob = Path(dir_path).glob("*.jpg")
    posters, genres, ids = [], [], []
    if verbose:
        path_list = tqdm(sorted(image_glob))
    else:
        path_list = sorted(image_glob)
    for path in path_list:
        index = get_id(path)
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
