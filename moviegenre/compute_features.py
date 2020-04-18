from pathlib import Path
import argparse
import yaml
from utils.histo import histo_RGB, show_histo
from utils.misc import list_to_date, triplet_to_str, read_csv_with_genres, create_logger, numpy_image_to_cv2
from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path
import argparse
from preprocessing.database import clean_database, download_database
from preprocessing.sets import preprocess_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pandas as pd

def main(args):
    #J'ai repris la base du début du main pour pouvoir le réintégrer si besoin
    config = yaml.safe_load(open(args.config, encoding='utf-8'))
    # Naming files
    nb_genres = len(config['genres'])
    appendix_split = 's{}t{}_'.format(
        config['size_per_genre'],
        config['testing_split']
        ) + triplet_to_str(config['image_size']) + '_' + str(nb_genres)

    model_name = config['model_name']+'_e{}b{}v{}_'.format(config['nb_epochs'], config['batch_size'], config['validation_split'])
    model_name += appendix_split
    logger = None   # create_logger(name=model_name, log_dir=Path(args.log_dir))
    selection_name = args.csv+'clean_poster_data_'+str(nb_genres)+'.csv'
    model_name = args.models_dir + model_name + '.h5'
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
                                  verbose=args.verbose, logger=logger)

    data_name = [Path(prefix+appendix_split) for prefix in [args.sets_dir + 'xtr_',
                                                            args.sets_dir + 'ytr_',
                                                            args.sets_dir + 'idtr_',
                                                            args.sets_dir + 'xtest_',
                                                            args.sets_dir + 'ytest_',
                                                            args.sets_dir + 'idtest_']]

    if data_name[0].exists() and data_name[1].exists() and data_name[2].exists() and data_name[3].exists() and data_name[4].exists() and data_name[5].exists():
        if args.verbose:
            print('Training and testing sets alreadey made')
        train_posters, train_genres, train_ids = np.load(data_name[0]), np.load(data_name[1]), np.load(data_name[2])
        test_posters, test_genres, test_ids = np.load(data_name[3]), np.load(data_name[4]), np.load(data_name[5])

    else:
        train_posters, train_genres, train_ids, test_posters, test_genres, test_ids = preprocess_data(
            clean_movies, config['genres'], config['size_per_genre'], args.posters, config['image_size'],
            config['seed'], testing_split=config['testing_split'], verbose=args.verbose, logger=logger
        )
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

    # Pour tester la conversion de numpy vers cv2
    if False:
        BRG_poster = numpy_image_to_cv2(train_posters[0])
        cv2.imshow("test", BRG_poster)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    histo_r_train = np.zeros((train_posters.shape[0], 256))
    histo_g_train = np.zeros((train_posters.shape[0], 256))
    histo_b_train = np.zeros((train_posters.shape[0], 256))

    if args.verbose :
        print("Computing the RGB histograms on the train set")

    for i, poster in tqdm(enumerate(train_posters)):
        BRG_poster = numpy_image_to_cv2(poster)
        hist = histo_RGB(BRG_poster)
        histo_r_train[i] = hist['r'][:, 0]
        histo_g_train[i] = hist['g'][:, 0]
        histo_b_train[i] = hist['b'][:, 0]
#    print(histo_r_train, histo_g_train, histo_b_train)

    histo_r_test = np.zeros((test_posters.shape[0], 256))
    histo_g_test = np.zeros((test_posters.shape[0], 256))
    histo_b_test = np.zeros((test_posters.shape[0], 256))

    if args.verbose :
        print("Computing the RGB histograms on the train set")

    for i, poster in tqdm(enumerate(test_posters)):

        BRG_poster = numpy_image_to_cv2(poster)
        hist = histo_RGB(BRG_poster)
        histo_r_test[i] = hist['r'][:, 0]
        histo_g_test[i] = hist['g'][:, 0]
        histo_b_test[i] = hist['b'][:, 0]

    #pour tester
    if False:
        plt.imshow(train_posters[135])
        plt.show()
        plt.close()
        plt.plot(range(256), train_posters[135], 'r')
        plt.plot(range(256), train_posters[135], 'g')
        plt.plot(range(256), train_posters[135], 'b')
        plt.show()
        plt.close()

    features_name_train = [Path(prefix + 'train') for prefix in [args.features+'histo_r_',
                                                                args.features+'histo_g_',
                                                                args.features+'histo_b_']]

    features_name_test = [Path(prefix + 'test') for prefix in [args.features+'histo_r_',
                                                                args.features+'histo_g_',
                                                                args.features+'histo_b_']]

    if args.save:
        np.save(features_name_train[0], histo_r_train)
        np.save(features_name_train[1], histo_g_train)
        np.save(features_name_train[2], histo_b_train)

        np.save(features_name_test[0], histo_r_test)
        np.save(features_name_test[1], histo_g_test)
        np.save(features_name_test[2], histo_b_test)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the config file', default='./default_config.yml')
    parser.add_argument('--log-dir', help='Path to the log directory', default='../log/')
    parser.add_argument('--posters', help='Path to the posters', default='../data/posters/')
    parser.add_argument('--models-dir', help='Path to the saved models', default='../data/models/')
    parser.add_argument('--database', help='Path to the databse csv', default='../data/poster_data.csv')
    parser.add_argument('--csv', help='Path to the clean csv', default='../data/')
    parser.add_argument('--sets-dir', help='Path to the training and testing sets', default='../data/sets/')
    # je rajoute un argument pour la direction des features
    parser.add_argument('--features', help='Path to the clean csv', default='../data/features/')

    parser.add_argument('-s', '--save', help='Save model', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args, _ = parser.parse_known_args()
    main(args)
