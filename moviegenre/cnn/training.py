# -*- coding: utf-8 -*-
import numpy as np
from cnn.model import create_cnn_v1


def train_model(
        training_posters, training_genres,
        nb_genres, image_size,
        nb_epochs, batch_size, validation_split,
        verbose=True, logger=None
):

    model = create_cnn_v1(nb_genres, image_size)

    class_weights = np.ones(nb_genres) / training_genres.sum(axis=0)
    class_weights = dict(enumerate(class_weights))

    training_history = model.fit(
        training_posters, training_genres,
        batch_size=batch_size, epochs=nb_epochs, validation_split=validation_split,
        class_weight=class_weights,
        verbose=verbose)

    return model, training_history
