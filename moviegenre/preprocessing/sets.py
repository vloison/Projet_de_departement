# -*- coding: utf-8 -*-
"""Preprocesses the input data"""
from pathlib import Path
from skimage import transform
import imageio
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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


def split_database(movies, genre_dict, size_per_genre, seed, testing_split=0.15):
    nb_genres = len(genre_dict)
    ids = np.empty(size_per_genre*nb_genres, dtype=int)
    genres = np.empty(size_per_genre*nb_genres, dtype=int)

    for i, row in movies.iterrows():
        movies.at[i, 'genre'] = genre_dict[row.genre]

    for i, genre in enumerate(movies.genre.unique()):
        select = movies[movies['genre'] == genre].sample(size_per_genre, random_state=seed)
        ids[i*size_per_genre:(i+1)*size_per_genre] = select.allocine_id.to_numpy()
        genres[i*size_per_genre:(i+1)*size_per_genre] = select.genre.to_numpy()
    return train_test_split(ids, genres, test_size=testing_split, random_state=seed, stratify=genres)


def int_to_cat_vect(n, size):
    x = np.zeros(size)
    x[n] = 1
    return x


def preprocess_data(movies, genres_dict, size_per_genre,  posters_path,
                    image_size, seed, testing_split=0.15,
                    verbose=True, logger=None):
    """Generates the data to be used by the neural network"""
    nb_genres = len(genres_dict)
    posters_path = Path(posters_path)
    train_ids, test_ids, train_genres, test_genres = split_database(
        movies, genres_dict, size_per_genre, seed, testing_split=testing_split)

    train_genres = np.array(list(map(
        lambda n: int_to_cat_vect(n, nb_genres), train_genres)))
    test_genres = np.array(list(map(
        lambda n: int_to_cat_vect(n, nb_genres), test_genres)))

    train_posters = np.zeros((
        len(train_ids), image_size[0], image_size[1], image_size[2]))
    generator = range(len(train_ids))
    to_delete = []
    if verbose:
        print('Generating training set...')
        generator = tqdm(generator)
    for i in generator:
        path = posters_path/(str(train_ids[i])+'.jpg')
        try:
            train_posters[i] = normalize(imageio.imread(path), image_size)
        except Exception as e:
            if verbose:
                print("Erreur {} with film {}".format(e, train_ids[i]))
            to_delete.append(i)
    train_posters = np.delete(train_posters, to_delete, 0)
    train_genres = np.delete(train_genres, to_delete, 0)
    train_ids = np.delete(train_ids, to_delete)

    test_posters = np.zeros((
        len(test_ids), image_size[0], image_size[1], image_size[2]))
    generator = range(len(test_ids))
    to_delete = []
    if verbose:
        print('Generating testing set...')
        generator = tqdm(generator)
    for i in generator:
        path = posters_path/(str(test_ids[i])+'.jpg')
        try:
            test_posters[i] = normalize(imageio.imread(path), image_size)
        except Exception as e:
            if verbose:
                print("Erreur {} with film {}".format(e, test_ids[i]))
            to_delete.append(i)
    test_posters = np.delete(test_posters, to_delete, 0)
    test_genres = np.delete(test_genres, to_delete, 0)
    test_ids = np.delete(test_ids, to_delete)
    return train_posters, train_genres, train_ids, test_posters, test_genres, test_ids
