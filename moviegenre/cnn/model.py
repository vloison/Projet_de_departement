# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adagrad
from utils.constants import GENRES_DICT, SIZE

def create_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(SIZE[0], SIZE[1], 3)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(GENRES_DICT), activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=Adagrad(), metrics=["accuracy"])

    return model
