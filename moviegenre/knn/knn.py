import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# # BUILDING THE DICTIONNARY OF FEATURES
# def dico_features(histo_type):
#     return dico_train, dico_test


class Observations:
    """ Classe qui permet de gérer les observations (une observation = un poster)
    et notamment de concilier les features qui sont de l'ordre des floats et celles qui sont des histogrammes

    Parameters
    ----------
    histo_dist : int
        distance utilisée pour les histogrammes l'entier renvoie à la méthode utilisée par la fonction cv2. compareHist
            - 1 -> distance du chi2 : CV_COMP_CHISQR
            - 3 -> distance de Bhattacharyya : CV_COMP_BHATTACHARYYA

    Attributes
    ----------
    n_observations : type
        nombre d'observations
    observations : type
        matrice répertoriant les données des observations (une ligne=une observation)
    distance : type
        fonction qui permet de calculer la distance entre 2 observations en prenant compte de la distance entre les histogrammes, de la position des histogrammes ...
    features_type : type
        dictionnaire qui répertorie la position des features qui correspondent à histogrammes dans les observations (en effet toutes les observations sont concaténées horizontalement)
        sous la forme de tuples (début, fin) et qui répertorie aussi les emplacementsdes features qui sont de simples float.
    histo_dist : int
        cf au dessus
    """

    def __init__(self, histo_dist):
        """initialisation de la classe

        Parameters
        ----------
        histo_dist : int
            distance utilisée pour les histogrammes l'entier renvoie à la méthode utilisée par la fonction cv2. compareHist
            - 1 -> distance du chi2 : CV_COMP_CHISQR
            - 3 -> distance de Bhattacharyya : CV_COMP_BHATTACHARYYA

        """
        self.n_observations = 0
        self.observations = None
        self.distance = None
        self.features_type = {'histo': [], 'float': []}
        self.histo_dist = histo_dist

    def add_histo_feature(self, features_matrix):
        """Permet d'ajouter une features de type histogramme aux observations

        Parameters
        ----------
        features_matrix : np.array
            matrice qui contient les histogrammes pour chaque poster (une ligne = un poster, une colonne = une bin)

        """

        if self.n_observations == 0:
            self.n_observations = features_matrix.shape[0]
            self.observations = features_matrix

        self.observations = np.concatenate((self.observations, features_matrix), axis=1)

        self.features_type['histo'].append((self.observations.shape[1] - features_matrix.shape[1], self.observations.shape[1]))

    def add_float_feature(self, features_list):
        """Permet d'ajouter une features de type histogramme aux observations

        Parameters
        ----------
        features_list : np.array,
            vecteur colonne qui contient la feature pour chaque poster (une ligne = un poster)

        """
        if self.n_observations == 0:
            self.n_observations = len(features_list)
            self.observations = np.array(features_list).reshape(self.n_observations, 1)

        self.observations = np.concatenate((self.observations, np.array(features_list).reshape(self.n_observations, 1)), axis=1)

        self.features_type['float'].append(self.observations.shape[1] - 1)

    def compute_distance(self):
        """calcule la distance qui permet de comparer deux observations en tenant
        compte des features histogrammes, de la distance entre histogrammes, mais aussi des features float
        la distance est ensuite répertoriée dans l'attribut distance

        """
        def aux(x1, x2):

            res = 0
            for begin, end in self.features_type['histo']:
                res += cv2.compareHist(x1[begin:end].astype('float32'), x2[begin:end].astype('float32'), self.histo_dist)

            for ind in self.features_type['float']:
                res += np.abs(x1[ind] - x2[ind])

            return res

        self.observations = self.observations.astype(float)
        self.distance = lambda x1, x2: aux(x1, x2)


# KNN FUNCTION AND RESULTS

def KNN(dataset, Xtr, tr_features, Ytr, training_ids, Xtest, test_features, testing_ids, ind, k, metric=None, print_results=False):
    """Calculates the genre of movie of indice ind in the testing set using a k-NN approach.

    Parameters
    ----------
    dataset : type
        Description of parameter `dataset`.
    Xtr : type
        set of posters among which the closest neighbors will be found
    tr_features : dictionnary of np.array of floats, depending on the feature
         dictionnary which keys are feature names, and the values are numpy arrays containing for each line, the feature corresponding to the poster in Xtr
    Ytr : type
        stocks the genre of each poster of Xtr`.
    training_ids : type
        stocks the indice of each poster of Xtr in the original database
    Xtest : type
        test posters
    test_features : type
         dictionnary which keys are feature names, and the values are numpy arrays containing for each line, the feature corresponding to the poster in Xtest
    testing_ids : type
        stocks the indice of each poster of Xtr in the original database
    ind : int
        ???
    k : int
        number of neighbours
    print_results : Bool
        Description of parameter `print_results`.
    metric : callable
        functions that calculates the distance between two_observations

    Returns
    -------
    type
        Description of returned object.

    """

    # Create k-NN operator
    print('Generating kNN Classifier...')
    if metric is not None:
        neigh = KNeighborsClassifier(n_neighbors=k, metric=metric)
    else:
        neigh = KNeighborsClassifier(n_neighbors=k)

    neigh.fit(tr_features, Ytr)
    print('kNN Classifier generated.')
    # Si ind > 0,renvoyer la prédiction pour test_features[ind] et afficher ses plus proches voisins
    if ind > 0:
    # Preprocess the poster to be classified
        x = Xtest[ind]
        if print_results:
            print("Movie to be labellised:")
            plt.imshow(x)
            plt.show()
        # Classification
        x_feat = test_features[ind]

        neighbors = neigh.kneighbors([x_feat], return_distance=False)
        label_for_max = Ytr[neighbors]
        label_for_max = np.sum(label_for_max, axis=1)
        print('labels_for_max', label_for_max)
        ind_genre = np.argmax(label_for_max)
        prediction = np.zeros(7)
        prediction[ind_genre] = 1
        print('Prediction:', prediction)
        #genre_ind = np.where(neigh.predict([x])[0] == 1)[0]
        #print('genre_ind', genre_ind)
        #genre = genres_dict[list(genre_ind)]
        # Print results
        if print_results:
            titre = dataset.loc[dataset['allocine_id'] == testing_ids[ind], ['title']].values[0]
            print("Title of the movie:", titre)
            print("Label: ", dataset.loc[dataset['allocine_id'] == testing_ids[ind], ['genre']].values[0])
            #print("Prediction for poster : ", genre_ind)
            # DIsplay neighbors


            print("Closest neighbors:")
            for neighbor in neighbors[0]:
                plt.imshow(Xtr[neighbor])
                plt.show()
                titre = dataset.loc[dataset['allocine_id'] == training_ids[neighbor], ['title']].values[0]
                label = dataset.loc[dataset['allocine_id'] == training_ids[neighbor], ['genre']].values[0]
                print("Title of the movie:", titre)
                print("Label: ", label)
        return prediction

    #Si ind = -1, renvoyer le vecteur des prédictions sur tout le testing set
    if ind ==-1:
        test_prediction = np.zeros((len(testing_ids), Ytr.shape[1]))
        for i in tqdm(range(len(testing_ids))):
            x_feat = test_features[i]

            neighbors = neigh.kneighbors([x_feat], return_distance=False)
            label_for_max = Ytr[neighbors]
            label_for_max = np.sum(label_for_max, axis=1)
            #print('labels_for_max', label_for_max)
            ind_genre = np.argmax(label_for_max)
            test_prediction[i][ind_genre] = 1

        return(test_prediction)


def test_KNN(dataset, Xtr, tr_features, Ytr, training_ids, Xtest, test_features, Ytest, testing_ids, k, accuracy_funct, genres_dict):
    # Initialize k-NN parameters
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(tr_features, Ytr)
    indecis = 0
    accuracy = 0
    for i in tqdm(range(len(testing_ids))):
        # Compute x
        x = test_features[i]
        # Prediction
        neighbors = neigh.kneighbors([x], return_distance=False)
        label_for_max = np.sum(Ytr[neighbors], axis=1)
        ind_genre = np.argmax(label_for_max)
        prediction = np.zeros(Ytr.shape[1])
        prediction[ind_genre] = 1
        if np.argmax(prediction) == 0:
            indecis +=1
        # Compute accuracy
        accuracy += accuracy_funct(Ytest[i], prediction)
    accuracy = accuracy/len(testing_ids)
    print('Nombre de cas indécis', indecis)
    return(accuracy)
