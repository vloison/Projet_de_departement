# -*- coding: utf-8 -*-
from cnn.model import create_cnn_v1


def train_model(
        training_posters, training_genres,
        nb_genres, image_size,
        nb_epochs, batch_size, validation_split,
        verbose=True, logger=None
):

    model = create_cnn_v1(nb_genres, image_size)
    model.fit(training_posters, training_genres, batch_size=batch_size, epochs=nb_epochs,
              verbose=verbose, validation_split=validation_split)
    return model
