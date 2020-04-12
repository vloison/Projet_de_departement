# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.metrics import categorical_accuracy


def standard_layer(conv1_dim, conv2_dim, input):
    output = Conv2D(conv1_dim, kernel_size=(3, 3), activation="relu")(input)
    output = Conv2D(conv2_dim, kernel_size=(3, 3), activation="relu")(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Dropout(0.25)(output)

    return output


def create_cnn_v1(nb_genres, size):
    input_poster = Input(shape=size)

    output = standard_layer(32, 64, input_poster)

    output = standard_layer(128, 64, output)

    output = Flatten()(output)
    output = Dense(128, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(nb_genres, activation="sigmoid")(output)

    model = Model(inputs=[input_poster], outputs=output)

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adagrad(),
        metrics=["accuracy", categorical_accuracy]
    )

    return model

