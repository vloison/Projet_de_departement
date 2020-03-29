"""Preprocesses the input data"""
from pathlib import Path
import imageio
import skimage.transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



def read_csv_with_genres(file_name):
    def aux(string):
        return ''.join(c for c in string if c not in ['[', ']', "'"])
    movies = pd.read_csv(file_name, index_col='allocine_id')
    movies['genres'] = movies['genres'].str.translate("[]'")
    for _, row in movies.iterrows():
        row.genres =  aux(row.genres).split(" ")
    return movies


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
    image_glob = Path(dir_path).glob("*.jpg")
    posters, genres, ids = [], [], []
    for path in tqdm(sorted(image_glob)):
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
            print("Erreur", e)
    if save:
        np.save('../data/numpy_posters.npy', posters)
        np.save('../data/numpy_genres', genres)
        np.save('../data/numpy_ids', ids)
    print('Done.')
    return posters, genres, ids


SAVELOCATION = '../data/posters/'
MOVIES = read_csv_with_genres('../data/clean_poster_data.csv')

GENRES_DICT = {
    'Action': 0,
    'Animation': 1,
    'Aventure': 2,
    'Biopic': 3,
    'Comédie': 4,
    'Comédie dramatique': 5,
    'Comédie musicale': 6,
    'Documentaire': 7,
    'Drame': 8,
    'Epouvante-horreur': 9,
    'Fantastique': 10,
    'Historique': 11,
    'Policier': 12,
    'Romance': 13,
    'Science fiction': 14,
    'Thriller': 15,
    'Western': 16
}

if __name__ == "__main__":
    X, Y, IDS = prepare_data(SAVELOCATION, MOVIES)
    show_img(MOVIES, X, Y, IDS, 13)
