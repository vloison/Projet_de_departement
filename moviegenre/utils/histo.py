# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


def histo_RGB(img):
    """Computes color histogram RGB of an image

    Parameters
    ----------
    img : type
        cv2 image

    Returns
    -------
    type
        Dictionnary containing the histograms, its key are 'b' (for blue), 'g' (green) or 'r' (red)

    """

    color = ('b', 'g', 'r')
    histo = {}

    for i, col in enumerate(color):
        histo[col] = cv2.calcHist([img],[i], None, [256], [0,255])

    return histo

def histo_LAB(img):
    """Computes color histogram LAB of an image

    Parameters
    ----------
    img : type
        cv2 image

    Returns
    -------
    type
        Dictionnary containing the histograms, its key are 'l' (for luminance), 'a' (green) or 'b' ()

    """

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    color = ('l', 'a', 'b')
    histo = {}

    for i, col in enumerate(color):
        histo[col] = cv2.calcHist([lab_img],[i], None, [256], [0,255])

    return histo

def show_histo_RGB(histo):

    plt.plot(range(256), histo['r'], 'r')
    plt.plot(range(256), histo['g'], 'g')
    plt.plot(range(256), histo['b'], 'b')
    plt.grid(True)
    plt.show()
