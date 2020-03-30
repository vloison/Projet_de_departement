# -*- coding: utf-8 -*-
"""Downloads the posters and cleans the database"""
from utils.constants import SAVELOCATION, MOVIES_PATH, CLEAN_MOVIES_PATH, FIRST_DATE, LAST_DATE

from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
from tqdm import tqdm  # barre de chargement
import requests
import numpy as np



def url_exist(dataset, verbose=True):
    not_found = []
    length = len(dataset)
    for index, row in tqdm(dataset.iterrows(), total=length):
        r = requests.head(row.poster)
        if r.status_code != requests.codes.ok:
            not_found.append(index)
    if verbose:
        print('URLS NOT FOUND:', not_found)
    return not_found


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
                urlretrieve(row.poster, str(Path(jpgname)))
        except Exception as e:
            print(e)
            not_found.append(index)
    print('Database downloaded')
    print(not_found)
    return not_found


def genre_count(movies, genre_column):
    """Prints the genres (first category) and the number of appearences"""
    for label in movies[genre_column].unique():
        occurences = len(movies[movies[genre_column] == label])
        print(label, occurences)
    print(len(movies[genre_column].unique()), '\n')


def replace_genres(movies):
    """Replace genres"""
    movies = movies.drop(movies[movies['genre_1'].isin(
        ['Classique', 'Concert', 'Opera', 'Famille', 'Divers', 'Erotique',
         'Sport event', 'Expérimental'])].index)
    for genre_column in ['genre_1', 'genre_2', 'genre_3']:
        movies.loc[movies[genre_column] == 'Dessin animé',
                   genre_column] = 'Animation'
        movies.loc[movies[genre_column] == 'Espionnage',
                   genre_column] = 'Thriller'
        movies.loc[movies[genre_column] == 'Musical',
                   genre_column] = 'Comédie musicale'
        movies.loc[movies[genre_column] == 'Péplum',
                   genre_column] = 'Aventure'
        movies.loc[movies[genre_column] == 'Judiciaire',
                   genre_column] = 'Thriller'
        movies.loc[movies[genre_column] == 'Bollywood',
                   genre_column] = 'Comédie musicale'
        movies.loc[movies[genre_column] == 'Arts Martiaux',
                   genre_column] = 'Action'
        movies.loc[movies[genre_column] == 'Guerre',
                   genre_column] = 'Action'
        movies.loc[movies[genre_column].isin([
            'Classique', 'Concert', 'Opera', 'Famille', 'Show', 'Divers',
            'Erotique', 'Sport event', 'Expérimental', 'Movie night']),
                   genre_column] = np.nan

    for genre_column in ['genre_1', 'genre_2', 'genre_3']:
        movies.loc[movies[genre_column] == 'Comédie dramatique',
                   genre_column] = 'Comédie-dramatique'
        movies.loc[movies[genre_column] == 'Comédie musicale',
                   genre_column] = 'Comédie-musicale'
        movies.loc[movies[genre_column] == 'Science fiction',
                   genre_column] = 'Science-fiction'
    return movies


def prepare_dataset(raw_movies, verbose=True):
    """Prepare dataset"""
    movies = raw_movies[['title', 'genre_1', 'genre_2', 'genre_3',
                         'release_date', 'pays', 'poster']].dropna(
                        subset=['title', 'genre_1', 'poster', 'release_date'])
    movies.drop(movies[LAST_DATE < movies['release_date'].map(
        pd.Timestamp)].index, inplace=True)
    movies.drop(movies[FIRST_DATE > movies['release_date'].map(
        pd.Timestamp)].index, inplace=True)

#     NOT_FOUND = url_exist(movies, verbose)
#     movies = movies.drop(movies.index[NOT_FOUND])
    movies = replace_genres(movies)
    movies['genres'] = movies[['genre_1', 'genre_2', 'genre_3']].apply(
        lambda x: ';'.join(x.dropna().astype(str)),
        axis=1
    ).str.split(';')
    genre_count(movies, 'genre_1')
    genre_count(movies, 'genre_2')
    genre_count(movies, 'genre_3')
    for _, row in movies.iterrows():
        row.genres = np.unique(row.genres)
    movies.drop(['genre_1', 'genre_2', 'genre_3'], axis=1, inplace=True)
    return movies


if __name__ == '__main__':
    RAW_MOVIES = pd.read_csv(MOVIES_PATH, sep=',',
                         index_col='allocine_id')
    MOVIES = prepare_dataset(RAW_MOVIES)
    MOVIES.to_csv(CLEAN_MOVIES_PATH)
    #database_download(SAVELOCATION, MOVIES)
