import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from keras_yolo.yolo import YOLO

def detect_image_dictionnary(yolo, image):
    """Short summary.

    Parameters
    ----------
    yolo : YOLO
        class defined in YOLO.py
    image : image

    Returns
    -------
    type
        A dictionnary containing the detected objects, their probability and their surronding boxes

    """
    found = yolo.detect_image_dictionnary(image)
    return(found)

def detect_image_draw(yolo, image):
    """Short summary.

    Parameters
    ----------
    yolo : YOLO
        class defined in YOLO.py
    image : image

    Returns
    -------
    type
        The image annoted with the frames of the detected objects

    """
    img = yolo.detect_image(image)
    return(img)

if __name__ == "__main__":
    path_image1 = "./keras_yolo/pictures/ete_chiot.jpeg"
    path_image2 = "./keras_yolo/pictures/TED.jpeg"


    yolo = YOLO()

    #retourne juste le dico

    image1 = Image.open(path_image1)
    found = detect_image_dictionnary(yolo, image1)
    print(found)
    

    #affiche l'image

    image2 = Image.open(path_image2)
    img = detect_image_draw(yolo, image2)
    plt.imshow(img)
    plt.show()

    yolo.close_session()
