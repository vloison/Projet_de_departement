# -*- coding: utf-8 -*-
"""Downloads the posters and cleans the database"""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
from tqdm import tqdm
import numpy as np


def genre_count(movies, genre_column):
    """Prints the genres (first category) and the number of appearences"""
    for label in movies[genre_column].unique():
        occurences = len(movies[movies[genre_column] == label])
        print(label, occurences)
    print(len(movies[genre_column].unique()), '\n')


def replace_genres(movies, genres):
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


def clean_database(movies_path, genres, start=None, end=None, verbose=True, logger=None):
    """Prepare dataset"""
    raw_movies = pd.read_csv(movies_path, sep=',',
                         index_col='allocine_id')
    movies = raw_movies[['title', 'genre_1', 'genre_2', 'genre_3',
                         'release_date', 'pays', 'poster']].dropna(
                        subset=['title', 'genre_1', 'poster', 'release_date'])
    if end is not None:
        movies.drop(movies[end < movies['release_date'].map(
        pd.Timestamp)].index, inplace=True)
    if start is not None:
        movies.drop(movies[start > movies['release_date'].map(
            pd.Timestamp)].index, inplace=True)
    movies = replace_genres(movies, genres)
    movies['genres'] = movies[['genre_1', 'genre_2', 'genre_3']].apply(
        lambda x: ';'.join(x.dropna().astype(str)),
        axis=1
    ).str.split(';')
    
    if verbose:
        genre_count(movies, 'genre_1')
        genre_count(movies, 'genre_2')
        genre_count(movies, 'genre_3')

    for _, row in movies.iterrows():
        row.genres = np.unique(row.genres)
    movies.drop(['genre_1', 'genre_2', 'genre_3'], axis=1, inplace=True)
    return movies


def download_database(savelocation, dataset, nb=None, verbose=True, logger=None):
    """Downloads the database from the given links in the dataset"""
    path = Path(savelocation)
    if not path.exists():
        path.mkdir()
    not_found = []
    if verbose:
        print('Posters database downloading')
    generator = dataset.iterrows() if nb is None else dataset.head(
        n=nb).iterrows()
    if verbose:
        length = len(dataset) if nb is None else nb
        generator = tqdm(generator, total=length)
    for index, row in generator:
        current_name = str(index)+'.jpg'
        jpgname = path / current_name
        try:
            if not Path(jpgname).is_file():
                urlretrieve(row.poster, str(Path(jpgname)))
        except Exception as e:
            if verbose:
                print('Error {} with film {}'.format(e, index))
            not_found.append(index)
    if verbose:
        print('Database downloaded')
    return not_found
