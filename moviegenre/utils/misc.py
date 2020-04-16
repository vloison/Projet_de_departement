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
    return  cv2.cvtColor(RGB_image, cv2.COLOR_RGB2BGR )


def create_logger(name, log_dir=None, debug=False):
    """Create a logger.
    Create a logger that logs to log_dir.
    Args:
        name: str. Name of the logger.
        log_dir: str. Path to log directory.
        debug: bool. Whether to set debug level logging.
    Returns:
        logging.logger.
    """

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(process)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format=log_format)
    logger = logging.getLogger(name)
    if log_dir:
        log_file = os.path.join(log_dir, '{}.txt'.format(name))
        file_hdl = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt=log_format)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    return logger
