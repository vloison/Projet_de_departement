# -*- coding: utf-8 -*-
import pandas as pd
import logging
import os
import numpy as np
import cv2


def read_csv_with_genres(file_name):
    def aux(string):
        return ''.join(c for c in string if c not in ['[', ']', "'"])
    movies = pd.read_csv(file_name, index_col='allocine_id')
    for _, row in movies.iterrows():
        row.genres =  aux(row.genres).split(" ")
    return movies


def list_to_date(l):
    return pd.Timestamp(year=l[0], month=l[1], day=l[2])


def triplet_to_str(l):
    return '{}-{}-{}'.format(l[0], l[1], l[2])



def parse_model_name(name):
    splitted = name.split(sep='_')
    splitted[3] = splitted[3].split(sep='-')
    b_index = splitted[1].rfind('b')
    v_index = splitted[1].rfind('v')
    t_index = splitted[2].rfind('t')

    return {
        'nn_version': splitted[0],
        'image_size': splitted[3],
        'nb_genres': splitted[4],
        'nb_epochs': int(splitted[1][1:b_index]),
        'batch_size': int(splitted[1][b_index+1:v_index]),
        'validation_split': float(splitted[1][v_index+1:])
    }


def numpy_image_to_cv2(RGB_image):
    """ As numpy uses RGB with pixel values in [0,1] and cv2 uses BRG with values in [0,256],
    this function converts a numpy image to a cv2 image

    Parameters
    ----------
    RGB_image : numpy image

    Returns
    -------
        np.array to be exploted by cv2

    """
    # BRG_image = np.zeros(RGB_image.shape, dtype=int)
    # BRG_image[:, :, 0] = RGB_image[:, :, 2]#np.floor(255 * RGB_image[:, :, 2])
    # BRG_image[:, :, 1] = RGB_image[:, :, 1]#np.floor(255 * RGB_image[:, :, 1])
    # BRG_image[:, :, 2] = RGB_image[:, :, 0]#np.floor(255 * RGB_image[:, :, 0])
    #
    # print("shape", BRG_image.shape,'\n')
    return  cv2.cvtColor((255 * RGB_image).astype('uint8'), cv2.COLOR_RGB2BGR )



