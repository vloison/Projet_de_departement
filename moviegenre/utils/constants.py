# -*- coding: utf-8 -*-
import pandas as pd

SAVELOCATION = '../data/posters/'
MOVIES_PATH = '../data/poster_data.csv'
CLEAN_MOVIES_PATH = '../data/clean_poster_data.csv'

GENRES_DICT = {
    'Action': 0,
    'Animation': 1,
    'Aventure': 2,
    'Biopic': 3,
    'Comédie': 4,
    'Comédie-dramatique': 5,
    'Comédie-musicale': 6,
    'Documentaire': 7,
    'Drame': 8,
    'Epouvante-horreur': 9,
    'Fantastique': 10,
    'Historique': 11,
    'Policier': 12,
    'Romance': 13,
    'Science-fiction': 14,
    'Thriller': 15,
    'Western': 16
}
SIZE = (150, 100, 3)
FIRST_DATE = pd.Timestamp(year=2010, month=1, day=1)
LAST_DATE = pd.Timestamp(year=2020, month=1, day=1)

TRAINING_SIZE = 2000  # Size of the training set
TESTING_SIZE = 100   # Size of the testing set
