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
from preprocessing.preprocessing import preprocess_data
from preprocessing.split import split_data
from cnn.training import train_model
from utils.accuracy import multi_label
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

def main(args):


    #J'ai repris la base du début du main pour pouvoir
    config = yaml.safe_load(open(args.config, encoding='utf-8'))
    # Naming files
    nb_genres = len(config['genres'])
    start = list_to_date(config['first_date'])
    end = list_to_date(config['last_date'])
    selection_name = triplet_to_str(config['first_date'])+'_'+triplet_to_str(config['last_date'])+'_'+str(nb_genres)
    appendix_data = triplet_to_str(config['image_size']) + '_' + selection_name
    appendix_split = config['split_method']+'_tr{}t{}_'.format(config['training_size'], config['testing_size'])+appendix_data
    model_name = config['model_name']+'_e{}b{}v{}_'.format(config['nb_epochs'], config['batch_size'], config['validation_split'])
    model_name += appendix_split
    logger = None #create_logger(name=model_name, log_dir=Path(args.log_dir))
    selection_name = args.csv + selection_name + '.csv'
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


    histo_r = np.zeros((posters.shape[0], 256))
    histo_g = np.zeros((posters.shape[0], 256))
    histo_b = np.zeros((posters.shape[0], 256))
    if args.verbose :
        print("Computing the RGB histograms")


    # plt.imshow(posters[140])
    # plt.show()
    # plt.close()
    #
    # cv2.imshow('img1', posters[140])
    # cv2.imshow('img', numpy_image_to_cv2(posters[140]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # plt.show(numpy_image_to_cv2(posters[1300]))

    for i, poster in tqdm(enumerate(posters)):
        BRG_poster = numpy_image_to_cv2(poster)
        hist = histo_RGB(BRG_poster)
        histo_r[i] = hist['r'][:, 0]
        histo_g[i] = hist['g'][:, 0]
        histo_b[i] = hist['b'][:, 0]
    print(histo_r, histo_g, histo_b)

    #pour tester
    if False:
        plt.imshow(posters[1300])
        plt.show()
        plt.close()
        plt.plot(range(256), histo_r[1300], 'r')
        plt.plot(range(256), histo_g[1300], 'g')
        plt.plot(range(256), histo_b[1300], 'b')
        plt.show()
        plt.close()

    features_name = [Path(prefix+appendix_data) for prefix in [args.features+'histo_r_', args.features+'histo_g_', args.features+'histo_b_']]

    if args.save:
        np.save(features_name[0], histo_r)
        np.save(features_name[1], histo_g)
        np.save(features_name[2], histo_b)

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
    # je rajoute un argument pour la direction des features
    parser.add_argument('--features', help='Path to the clean csv', default='../data/features/')


    parser.add_argument('-s', '--save', help='Save model', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args, _ = parser.parse_known_args()
    main(args)
