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
    generator = dataset.iterrows() if nb is None else dataset.head(
        n=nb).iterrows()
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
TODAY = pd.Timestamp(year=2020, month=3, day=10)

MOVIES = RAW_MOVIES.dropna(
    subset=['title', 'release_date', 'genre_1', 'poster']).drop(
        RAW_MOVIES[RAW_MOVIES['release_date'].map(pd.Timestamp) > TODAY].index)
# MOVIES.profile_report()


def genre_count(movies):
    """Prints the genres (first category) and the number of appearences"""
    genre_list = movies.genre_1.unique()
#     for genre in np.append(movies.genre_2.unique(), movies.genre_3.unique()):
#         if genre not in genre_list:
#             np.append(genre_list, genre)
    for label in genre_list:
        occurences = len(movies[movies['genre_1'] == label])
        print(label, occurences)


MOVIES = MOVIES.drop(MOVIES[MOVIES['genre_1'].isin(
    ['Classique', 'Concert', 'Opera', 'Famille', 'Divers', 'Erotique',
     'Sport event', 'Expérimental'])].index)
MOVIES.loc[MOVIES['genre_1'] == 'Dessin animé', 'genre_1'] = 'Animation'
MOVIES.loc[MOVIES['genre_1'] == 'Espionnage', 'genre_1'] = 'Thriller'
MOVIES.loc[MOVIES['genre_1'] == 'Musical', 'genre_1'] = 'Comédie musicale'
MOVIES.loc[MOVIES['genre_1'] == 'Péplum', 'genre_1'] = 'Aventure'
MOVIES.loc[MOVIES['genre_1'] == 'Judiciaire', 'genre_1'] = 'Thriller'
MOVIES.loc[MOVIES['genre_1'] == 'Bollywood', 'genre_1'] = 'Comédie musicale'
MOVIES.loc[MOVIES['genre_1'] == 'Arts Martiaux', 'genre_1'] = 'Action'
MOVIES.loc[MOVIES['genre_1'] == 'Guerre', 'genre_1'] = 'Action'

genre_count(MOVIES)

NOT_FOUND = database_download(SAVELOCATION, MOVIES)
MOVIES = MOVIES.drop(MOVIES.index[NOT_FOUND])
MOVIES.to_csv('../data/clean_poster_data.csv', index=True)
