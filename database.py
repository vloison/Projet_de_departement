"""Downloads the posters and cleans the database"""
import os
from urllib.request import urlretrieve
import pandas as pd
from tqdm import tqdm  # barre de chargement


def database_download(savelocation, dataset):
    """Downloads the database from the given links in the dataset"""
    if not os.path.exists(savelocation):
        os.mkdir(savelocation)
    not_found = []
    posters_urls = dataset['poster'].to_list()
    print('Posters database downloading')
    for index in tqdm(range(len(posters_urls))):
        jpgname = savelocation+str(index)+'.jpg'
        try:
            if not os.path.isfile(jpgname):
                urlretrieve(posters_urls[index], jpgname)
        except Exception:  # Mettre le nom de l'exception possible
            not_found.append(index)
    print('Database downloaded')
    return not_found


SAVELOCATION = 'posters/'
RAW_MOVIES = pd.read_csv('poster_data.csv')
TODAY = pd.Timestamp.today()

'''
On nettoie la base de données en ne gardant que les films ayant un titre,
une date de sortie, au moins un genre et un poster.
On enlève les films pas encore sortis.
'''
MOVIES = RAW_MOVIES.dropna(
    subset=['title', 'release_date', 'genre_1', 'poster']).drop(
        RAW_MOVIES[RAW_MOVIES['release_date'].map(pd.Timestamp) > TODAY].index)


def genre_count(movies):
    """Prints the genres (first category) and the number of appearences"""
    genre_list = movies.genre_1.unique()
#     for genre in np.append(movies.genre_2.unique(), movies.genre_3.unique()):
#         if genre not in genre_list:
#             np.append(genre_list, genre)
    for label in genre_list:
        occurences = len(movies[movies['genre_1'] == label])
        print(label, occurences)


genre_count(MOVIES)
'''
Voir quels genres supprimer
Choisir un échantillon d'entraînement à partir des films restants
Pas forcément tout faire au hasard: on a un training set où certains genres
sont bien plus représentés.
'''
MOVIES.to_csv('clean_poster_data.csv')
# database_download(savelocation, movies)
