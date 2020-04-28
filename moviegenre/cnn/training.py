# -*- coding: utf-8 -*-
from pathlib import Path
from cnn.model import create_cnn_v1, create_cnn_v2, create_resnet
from utils.misc import parse_model_name
from tensorflow.keras.models import load_model


def train_model(
        version, training_posters, training_genres,
        nb_genres, image_size,
        nb_epochs, batch_size, validation_split,
        verbose=True, logger=None
):
    if version == 'cnn_v1':
        model = create_cnn_v1(nb_genres, image_size)
    else:
        model = create_resnet(nb_genres, image_size)

    training_history = model.fit(
        training_posters, training_genres,
        batch_size=batch_size, epochs=nb_epochs, validation_split=validation_split,
        verbose=verbose)

    return model, training_history



def get_trained_model(model_name, train_genres=None, train_posters=None, save_model=True, verbose=True):
    config = parse_model_name(model_name)
    if Path(model_name+'.h5').exists():
        if verbose:
            print('Model already trained')
        if Path('history_'+model_name+'.csv').exists():
            training_history = pd.read_csv(Path('history_'+model_name+'.csv'))
        else:
            if verbose:
                print('No training history')
                training_history = None
        return load_model(str(Path(model_name+'.h5'))), training_history
    model, training_history = train_model(
        config['nn_version'], train_posters, train_genres, config['nb_genres'], config['image_size'],
        nb_epochs=config['nb_epochs'], batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        verbose=verbose)
    if save_model:
        models_path = Path(config['models_dir'])
        if not models_path.exists():
            models_path.mkdir()
        model.save(str(Path(model_name+'.h5')))
        hist_df = pd.DataFrame(training_history.history)
        hist_df.to_csv(str(Path('history_'+model_name+'.csv')))
    return model, training_history
