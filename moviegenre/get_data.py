# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import yaml
from preprocessing.database import clean_database, download_database
from preprocessing.sets import preprocess_data
from utils.misc import triplet_to_str
import numpy as np
import pandas as pd


def main(args):
    config = yaml.safe_load(open(args.config, encoding='utf-8'))
    # Naming files
    nb_genres = len(config['genres'])
    appendix_split = 's{}t{}_'.format(
        config['size_per_genre'],
        config['testing_split']
    ) + triplet_to_str(config['image_size']) + '_' + str(nb_genres)

    selection_name = args.csv + 'clean_poster_data_' + str(nb_genres) + '.csv'
    model_name = args.models_dir + config['nn_version'] + '_e{}b{}v{}_'.format(
        config['nb_epochs'], config['batch_size'], config['validation_split'])
    model_name += appendix_split

    appendix_split += '.npy'

    if Path(selection_name).exists():
        if args.verbose:
            print('Database already cleaned')
        clean_movies = pd.read_csv(Path(selection_name))
    else:
        clean_movies = clean_database(Path(args.database))
        if args.save:
            clean_movies.to_csv(Path(selection_name))

    posters_path = Path(args.posters)
    if not posters_path.exists():
        posters_path.mkdir()
    not_found = download_database(posters_path, clean_movies,
                                  verbose=args.verbose)

    data_name = [Path(prefix + appendix_split) for prefix in [args.sets_dir + 'xtr_',
                                                              args.sets_dir + 'ytr_',
                                                              args.sets_dir + 'idtr_',
                                                              args.sets_dir + 'xtest_',
                                                              args.sets_dir + 'ytest_',
                                                              args.sets_dir + 'idtest_']]

    if data_name[0].exists() and data_name[1].exists() and data_name[2].exists() and data_name[3].exists() and data_name[4].exists() and data_name[5].exists():
        if args.verbose:
            print('Training and testing sets already made')
        train_posters, train_genres, train_ids = np.load(
            data_name[0]), np.load(data_name[1]), np.load(data_name[2])
        test_posters, test_genres, test_ids = np.load(
            data_name[3]), np.load(data_name[4]), np.load(data_name[5])

    else:
        train_posters, train_genres, train_ids, test_posters, test_genres, test_ids = preprocess_data(
            clean_movies, config['genres'], config['size_per_genre'], args.posters, config['image_size'],
            config['seed'], testing_split=config['testing_split'], verbose=args.verbose)
        if args.save:
            sets_path = Path(args.sets_dir)
            if not sets_path.exists():
                sets_path.mkdir()
            np.save(data_name[0], train_posters)
            np.save(data_name[1], train_genres)
            np.save(data_name[2], train_ids)
            np.save(data_name[3], test_posters)
            np.save(data_name[4], test_genres)
            np.save(data_name[5], test_ids)

    return clean_movies, train_posters, train_genres, train_ids, test_posters, test_genres, test_ids, model_name, args.save, args.verbose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', help='Path to the config file', default='./default_config.yml')
    parser.add_argument(
        '--posters', help='Path to the posters', default='../data/posters/')
    parser.add_argument(
        '--sets-dir', help='Path to the training and testing sets', default='../data/sets/')
    parser.add_argument(
        '--database', help='Path to the databse csv', default='../data/poster_data.csv')
    parser.add_argument(
        '--csv', help='Path to the clean csv', default='../data/')
    parser.add_argument('--models-dir', help='Path to the saved models', default='../data/models/')
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    parser.add_argument('-s', '--save', help='Save model', action='store_true')
    args, _ = parser.parse_known_args()
    main(args)
