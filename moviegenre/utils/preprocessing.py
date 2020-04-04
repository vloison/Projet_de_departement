# -*- coding: utf-8 -*-
import pandas as pd
from skimage import transform
import numpy as np

def read_csv_with_genres(file_name):
    def aux(string):
        return ''.join(c for c in string if c not in ['[', ']', "'"])
    movies = pd.read_csv(file_name, index_col='allocine_id')
    for _, row in movies.iterrows():
        row.genres =  aux(row.genres).split(" ")
    return movies


def get_id(path):
    """Gets the id from the Pathlib path"""
    filename = path.parts[-1]
    index_f = filename.rfind(".jpg")
    return int(filename[:index_f])


def normalize(img, size=(150, 100, 3)):
    """Normalizes the image"""
    img = transform.resize(img, size)
    img = img.astype(np.float32)
#     img = (img / 127.5) -1
    return img
