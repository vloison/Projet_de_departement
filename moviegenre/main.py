# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import yaml
from preprocessing.database import clean_database, download_database
from preprocessing.preprocessing import preprocess_data
from preprocessing.split import split_data
from cnn.training import train_model
from utils.accuracy import multi_label
from utils.misc import list_to_date, triplet_to_str, read_csv_with_genres, create_logger
from tensorflow.keras.models import load_model
import numpy as np


def main(args):
    config = yaml.safe_load(open(args.config, encoding='utf-8'))
    # Naming files
    nb_genres = len(config['genres'])
    start = list_to_date(config['first_date'])
    end = list_to_date(config['last_date'])
    selection_name = triplet_to_str(config['first_date'])+'_'+triplet_to_str(config['last_date'])+'_'+str(nb_genres)
    appendix_data = triplet_to_str(config['image_size']) + '_' + selection_name
    appendix_split = config['split_method']+'_tr{}t{}_'.format(config['training_size'], config['testing_size'])+appendix_data
    model_name = 'cnn1_e{}b{}v{}_'.format(config['nb_epochs'], config['batch_size'], config['validation_split'])
    model_name += appendix_split
    logger = None #create_logger(name=model_name, log_dir=Path(args.log_dir))
    selection_name = args.csv+selection_name+'.csv'
    appendix_data += '.npy'
    model_name = args.models_dir + model_name + '.h5'

    if Path(selection_name).exists():
        if args.verbose:
            print('Database already cleaned')
        selected_movies = read_csv_with_genres(Path(selection_name))
    else:
        selected_movies = clean_database(Path(args.database), config['genres'],
                                         start=start, end=end,
                                         verbose=args.verbose, logger=logger)
        if args.save:
            selected_movies.to_csv(Path(selection_name))
    not_found = download_database(Path(args.posters), selected_movies,
                                  verbose=args.verbose, logger=logger)

    data_name = [Path(prefix+appendix_data) for prefix in [args.preproc+'x_', args.preproc+'y_', args.preproc+'id_']]
    if data_name[0].exists() and data_name[1].exists() and data_name[2].exists():
        if args.verbose:
            print('Data already preprocessed')
        posters, genres, ids = np.load(data_name[0]), np.load(data_name[1]), np.load(data_name[2])
    else:
        posters, genres, ids = preprocess_data(Path(args.posters), selected_movies, config['genres'],
                                               config['image_size'], verbose=args.verbose, logger=logger)
        if args.save:
            np.save(data_name[0], posters)
            np.save(data_name[1], genres)
            np.save(data_name[2], ids)

    data_name = [Path(prefix+appendix_data) for prefix in [args.split+'xtr_', args.split+'ytr_', args.split+'idtr_', args.split+'xtest_', args.split+'ytest_', args.split+'idtest_']]
    if data_name[0].exists() and data_name[1].exists() and data_name[2].exists() and data_name[3].exists() and data_name[4].exists() and data_name[5].exists():
        if args.verbose:
            print('Data already splitted')
        train_posters, train_genres, train_ids, test_posters, test_genres, test_ids = np.load(data_name[0]), np.load(data_name[1]), np.load(data_name[2]), np.load(data_name[3]), np.load(data_name[4]), np.load(data_name[5])
    else:
        train_posters, train_genres, train_ids, test_posters, test_genres, test_ids = split_data(
            selected_movies, posters, genres, ids, config['genres'], config['training_size'], config['testing_size'],
            config['split_method'], verbose=args.verbose, logger=logger)
        if args.save:
            np.save(data_name[0], train_posters)
            np.save(data_name[1], train_genres)
            np.save(data_name[2], train_ids)
            np.save(data_name[3], test_posters)
            np.save(data_name[4], test_genres)
            np.save(data_name[5], test_ids)



    if Path(model_name).exists():
        if args.verbose:
            print('Model already trained')
        model = load_model(str(Path(model_name)))
    else:
        model = train_model(train_posters, train_genres, nb_genres, config['image_size'], nb_epochs=config['nb_epochs'],
                        batch_size = config['batch_size'], validation_split = config['validation_split'],
                        verbose=args.verbose, logger=logger)
        if args.save:
            model.save(str(Path(model_name)))

    predicted_genres = model.predict(test_posters)
    print(multi_label(test_genres, predicted_genres, logger=logger))
    return model, test_posters, test_genres, test_ids, selected_movies, predicted_genres



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the config file', default='./default_config.yml')
    parser.add_argument('--log-dir', help='Path to the log directory', default='../log/')
    parser.add_argument('--posters', help='Path to the posters', default='../data/posters/')
    parser.add_argument('--models-dir', help='Path to the saved models', default='../data/models/')
    parser.add_argument('--preproc', help='Path to the preprocessed data', default='../data/preprocessed/')
    parser.add_argument('--split', help='Path to the splitted data', default='../data/splitted/')
    parser.add_argument('--database', help='Path to the databse csv', default='../data/poster_data.csv')
    parser.add_argument('--csv', help='Path to the clean csv', default='../data/')
    parser.add_argument('-s', '--save', help='Save model', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args, _ = parser.parse_known_args()
    main(args)
