# -*- coding: utf-8 -*-
import numpy as np
from cnn.model import create_cnn_v1, create_cnn_v2
from cnn.resnet import create_resnet


def train_model(
        model_name, training_posters, training_genres,
        nb_genres, image_size,
        nb_epochs, batch_size, validation_split,
        verbose=True, logger=None
):
    if model_name == 'cnn_v1':
        model = create_cnn_v1(nb_genres, image_size)
    else:
        model = create_resnet(nb_genres, image_size)

    training_history = model.fit(
        training_posters, training_genres,
        batch_size=batch_size, epochs=nb_epochs, validation_split=validation_split,
        verbose=verbose)

    return model, training_history
