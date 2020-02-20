import pandas as pd
import numpy as np
import os
from tqdm import tqdm  # barre de chargement
from urllib.request import urlretrieve


def database_download(savelocation, dataset):
    if not os.path.exists(savelocation):
        os.mkdir(savelocation)
    not_found = []
    postersURLS = dataset['poster'].to_list()
    print('Posters database downloading')
    for index in tqdm(range(len(postersURLS))):
        jpgname = savelocation+str(index)+'.jpg'
        try:
            if not os.path.isfile(jpgname):
                urlretrieve(postersURLS[index], jpgname)
        except:
            not_found.append(index)
    print('Database downloaded')
    return not_found


savelocation = 'posters/'
raw_movies = pd.read_csv('poster_data.csv')
today = pd.Timestamp.today()

# On nettoie la base de données en ne gardant que les films ayant un titre,
# une date de sortie, au moins un genre et un poster.
# On enlève les films pas encore sortis.
movies = raw_movies.dropna(
    subset=['title', 'release_date', 'genre_1', 'poster']).drop(
        raw_movies[raw_movies['release_date'].map(pd.Timestamp) > today].index)
# movies.to_csv('out.csv')


def genre_count(movies):
    # Liste des genres répertoriés
    genre_list = movies.genre_1.unique()
#     for genre in np.append(movies.genre_2.unique(), movies.genre_3.unique()):
#         if genre not in genre_list:
#             np.append(genre_list, genre)
    for label in genre_list:
        occurences = len(movies[movies['genre_1'] == label])
        print(label, occurences)


genre_count(movies)
