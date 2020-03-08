"""Downloads the posters and cleans the database"""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
from tqdm import tqdm  # barre de chargement


def database_download(savelocation, dataset, nb=None):
    """Downloads the database from the given links in the dataset"""
    path = Path(savelocation)
    if not path.exists():
        path.mkdir()
    not_found = []
    print('Posters database downloading')
    generator = dataset.iterrows() if nb is None else dataset.head(n=nb).iterrows()
    length = len(dataset) if nb is None else nb
    for index, row in tqdm(generator, total=length):
        current_name = str(index)+'.jpg'
        jpgname = path / current_name
        try:
            if not Path(jpgname).is_file():
                urlretrieve(row.poster, Path(jpgname))
        except Exception:  # Mettre le nom de l'exception possible
            not_found.append(index)
    print('Database downloaded')
    return not_found


SAVELOCATION = '../data/posters/'
RAW_MOVIES = pd.read_csv('../data/poster_data.csv')
TODAY = pd.Timestamp.today()

'''
On nettoie la base de données en ne gardant que les films ayant un titre,
une date de sortie, au moins un genre et un poster.
On enlève les films pas encore sortis.
'''
MOVIES = RAW_MOVIES.dropna(
    subset=['title', 'release_date', 'genre_1', 'poster']).drop(
        RAW_MOVIES[RAW_MOVIES['release_date'].map(pd.Timestamp) > TODAY].index)
print(MOVIES.columns)
# MOVIES.profile_report()

def genre_count(movies):
    """Prints the genres (first category) and the number of appearences"""
    genre_list = movies.genre_1.unique()
#     for genre in np.append(movies.genre_2.unique(), movies.genre_3.unique()):
#         if genre not in genre_list:
#             np.append(genre_list, genre)
    compt = 0
    for  label in genre_list:
        occurences = len(movies[movies['genre_1'] == label])
        print(label, occurences)
        if occurences > 100:
            compt +=1
    print(compt)


genre_count(MOVIES)
'''
Voir quels genres supprimer
Choisir un échantillon d'entraînement à partir des films restants
Pas forcément tout faire au hasard: on a un training set où certains genres
sont bien plus représentés.
'''
#MOVIES.to_csv('../data/clean_poster_data.csv', index=True)
#not_found = database_download(SAVELOCATION, MOVIES)
#print(not_found)
