# -*- coding: utf-8 -*-
"""Downloads the posters and cleans the database"""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
from tqdm import tqdm


def genre_count(movies, genre_column):
    """Prints the genres (first category) and the number of appearences"""
    print('Nb de genres {}; Nb de films {}'.format(
        len(movies[genre_column].unique()), len(movies)))
    for label in movies[genre_column].unique():
        occurences = len(movies[movies[genre_column] == label])
        print(label, occurences)


def clean_database(movies_path, keep=['Action', 'Drame', 'Comédie dramatique',
                                      'Animation', 'Documentaire', 'Comédie',
                                      'Policier', 'Thriller'],
                   no_release_date=False):
    """Prepare dataset"""
    raw_movies = pd.read_csv(movies_path, sep=',')
    if no_release_date:
        movies = raw_movies[['title', 'release_date', 'poster', 'allocine_id',
                             'genre_1', 'genre_2', 'genre_3']].dropna(
                        subset=['title', 'genre_1', 'poster'])
    else:
        movies = raw_movies[['title', 'release_date', 'poster', 'allocine_id',
                             'genre_1', 'genre_2', 'genre_3']].dropna(
                        subset=['title', 'genre_1', 'poster', 'release_date'])

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
                   genre_column] = ''

    to_drop = []
    for index, row in movies.iterrows():
        if row.genre_1 not in keep:
            if row.genre_2 in keep:
                movies.at[index, 'genre_1'] = row.genre_2
            elif row.genre_3 in keep:
                movies.at[index, 'genre_1'] = row.genre_3
            else:
                to_drop.append(index)

    movies['genre'] = movies['genre_1']
    movies.drop(to_drop, inplace=True)
    movies.drop(['genre_1', 'genre_2', 'genre_3'], axis=1, inplace=True)

    movies.loc[movies['genre'].isin(
        ['Thriller', 'Policier']), 'genre'] = 'Thriller-Policier'
    return movies


def download_database(path, dataset, nb=None, verbose=True):
    """Downloads the database from the given links in the dataset"""
    not_found = []
    if verbose:
        print('Posters database downloading')
    generator = dataset.iterrows() if nb is None else dataset.head(
        n=nb).iterrows()
    if verbose:
        length = len(dataset) if nb is None else nb
        generator = tqdm(generator, total=length)
    for index, row in generator:
        current_name = str(row.allocine_id)+'.jpg'
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
