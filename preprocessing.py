"""Preprocesses the input data"""
from pathlib import Path
import imageio
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


SAVELOCATION = 'posters/'
MOVIES = pd.read_csv('clean_poster_data.csv', index_col=0)


def get_id(path):
    """Gets the id from the Pathlib path"""
    filename = path.parts[-1]
    index_f = filename.rfind(".jpg")
    return int(filename[:index_f])


def show_img(dataset, posters, labels, ids, index):
    """Shows the image with id at index position in ids"""
    title = dataset.at[ids[index], 'title']
    genre = dataset.at[ids[index], 'genre_1']+', '+str(ids[index])
#     genres = [dataset.at[ids[index],'genre_'+i] for i in ['1','2','3']]
    plt.imshow(posters[index])
    plt.title('{} \n {}'.format(title, genre))
    plt.show()


def preprocess(img, size=(150, 100, 3)):
    """Normalizes the image"""
    img = skimage.transform.resize(img, size)
    img = img.astype(np.float32)
#     img = (img / 127.5) -1
    return img


def prepare_data(dir_path, dataset, size=(150, 100, 3), save=True):
    """Generates the data to be used by the neural network"""
    print('Generating dataset...')
    genre_list = dataset.genre_1.unique()
    nb_genres = len(genre_list)
    inv_genre = {genre_list[k]: k for k in range(nb_genres)}
    image_glob = sorted(Path(dir_path).glob("*.jpg"))
    posters, genres, ids = [], [], []
    for path in tqdm(image_glob):
        try:
            # Meilleure gestion d'erreur à faire
            posters.append(preprocess(imageio.imread(path), size))
            index = get_id(path)
            vect_genre = np.zeros(nb_genres, dtype=int)
            # Rajoute un 1 à l'indice correspondant à la position
            # du premier genre de ce film dans la liste des genres
            vect_genre[inv_genre[dataset.at[index, 'genre_1']]] = 1
            genres.append(vect_genre)
            ids.append(index)
        except Exception as e:
            print(e)
    if save:
        np.save('numpy_posters', posters)
        np.save('numpy_genres', genres)
        np.save('numpy_ids', ids)
    print('Done.')
    return posters, genres, ids


# X, Y, IDS = prepare_data(SAVELOCATION, MOVIES)
X = np.load('numpy_posters.npy')
Y = np.load('numpy_genres.npy')
IDS = np.load('numpy_ids.npy')
show_img(MOVIES, X, Y, IDS, 13)
