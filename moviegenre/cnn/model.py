# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adagrad


def create_cnn_v1(nb_genres, size):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=size),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(nb_genres, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer=Adagrad(), metrics=["accuracy"])
    return model



#     kernel_dimensions1 = (3,3)
#     kernel_dimensions2 = (3,3)
#     model = Sequential([
#         Conv2D(32, kernel_dimensions1, padding='same', input_shape=SIZE, activation='relu'),
#         Conv2D(32, kernel_dimensions1, activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.25),
# 
#         Conv2D(64, kernel_dimensions2, padding='same', activation='relu'),
#         Conv2D(64, kernel_dimensions2, activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.25),
# 
#         Flatten(),
#         Dense(512, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='sigmoid')
#     ])
# 
#     opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
