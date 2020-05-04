# -*- coding: utf-8 -*-
from pathlib import Path
from cnn.model import create_cnn, create_resnet
from utils.misc import parse_model_name
from tensorflow import keras
import pandas as pd

def train_model(
        version, train_posters, train_genres,
        nb_genres, image_size,
        nb_epochs, batch_size, validation_split,
        verbose=True):
    if version == 'cnn1':
        if verbose:
            print('Building CNN first version')
        model = create_cnn(nb_genres, image_size)
    else:
        if verbose:
            print('Using transfer learning on ResNet50V2')
        model = create_resnet(nb_genres, image_size)

    training_history = model.fit(
        x=train_posters, y=train_genres,
        batch_size=batch_size, epochs=nb_epochs, validation_split=validation_split,
        verbose=verbose)

    return model, training_history


def get_trained_model(model_name, train_posters=None, train_genres=None, save_model=True, verbose=True):
    config = parse_model_name(model_name)
    if config['nn_version'] == 'onlyresnet':
        if verbose:
            print('Loading keras ResNet50V2')

        nb_removed_layers = 1

        resnet = keras.applications.resnet_v2.ResNet50V2(
            input_shape=config['image_size'], include_top=False, weights="imagenet"
        )

        return keras.models.Model(
            inputs=resnet.input,
            outputs=resnet.layers[-nb_removed_layers].output
        ), None

    if Path(model_name+'.h5').exists():
        if verbose:
            print('Model already trained')
        if Path('history_'+model_name+'.csv').exists():
            training_history = pd.read_csv(Path('history_'+model_name+'.csv'))
        else:
            if verbose:
                print('No training history')
                training_history = None
        return keras.models.load_model(str(Path(model_name+'.h5'))), training_history

    model, training_history = train_model(
        config['nn_version'], train_posters, train_genres, config['nb_genres'], config['image_size'],
        nb_epochs=config['nb_epochs'], batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        verbose=verbose)
    if save_model:
        if not config['models_dir'].exists():
            config['models_dir'].mkdir()
        model.save(str(Path(model_name+'.h5')))
        hist_df = pd.DataFrame(training_history.history)
        hist_df.to_csv(str(Path('history_'+model_name+'.csv')))
    return model, training_history
