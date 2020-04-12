# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def histo(img):
    """Computes color histogram of an image

    Parameters
    ----------
    img : type
        cv2 image

    Returns
    -------
    type
        Dictionnary containing the histograms, its key are 'b' (for blue), 'g' (green) or 'r' (red)

    """

    color = ('b','g','r')
    histo = {}

    for i,col in enumerate(color):
        histo[col] = cv2.calcHist([img],[i],None,[256],[0,256])

    print(histo)

if __name__=='__main__':
    img = cv2.imread('../../data/posters/133194.jpg')
    plt.close()
    histo_NB(img)
