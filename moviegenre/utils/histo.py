# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


def histo_RGB(img, bins=256):
    """Computes color histogram RGB of an image

    Parameters
    ----------
    img : cv2 image
        image to be treated

    bins : positive int (default : 256)
        number of bins of the histogram

    Returns
    -------
    type
        Dictionnary containing the histograms, its key are 'b' (for blue), 'g' (green) or 'r' (red)

    """

    color = ('b', 'g', 'r')
    histo = {}

    for i, col in enumerate(color):
        histo[col] = cv2.calcHist([img],[i], None, [bins], [0,255])
        print(len(histo[col]))

    return histo

def histo_LAB(img, bins=256):
    """Computes color histogram LAB of an image

    Parameters
    ----------
    img : cv2 image
        image to be treated

    bins : positive int (default : 256)
        number of bins of the histogram

    Returns
    -------
    type
        Dictionnary containing the histograms, its key are 'l' (for luminance), 'a' (green) or 'b' ()

    """

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    color = ('l', 'a', 'b')
    histo = {}

    for i, col in enumerate(color):
        histo[col] = cv2.calcHist([lab_img],[i], None, [bins], [0,255])

    return histo

def show_histo_RGB(histo):

    plt.plot(range(len(histo['r'])), histo['r'], 'r')
    plt.plot(range(len(histo['g'])), histo['g'], 'g')
    plt.plot(range(len(histo['b'])), histo['b'], 'b')
    plt.grid(True)
    plt.show()
