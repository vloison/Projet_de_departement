from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class KNN(KNeighborsClassifier):
    def __init__(self, n_neighbors, verbose=True, weights='distance'):
        super().__init__(n_neighbors=n_neighbors, weights=weights)
        self.verbose = verbose
        self.neighbors = None

    def predict(self, test_features, train_genres):
        self.neighbors = []
        self.distances = []
        generator = tqdm(range(len(test_features))) if self.verbose else range(len(test_features))
        predicted_genres = np.zeros((len(test_features), train_genres.shape[1]))
        if self.verbose:
            print('Predicting...')
        for i in generator:
            # Récupération des plus proches voisins et de leurs distances
            distances, neighbors = self.kneighbors([test_features[i]], return_distance=True)
            self.neighbors.append(neighbors)
            self.distances.append(distances)
            # Traitement pour la prédiction 
            label_for_max = train_genres[neighbors] / distances
            label_for_max = np.sum(label_for_max, axis=1)
            ind_genre = np.argmax(label_for_max)
            predicted_genres[i][ind_genre] = 1
        # Mise à jour de self.neighbors et self.distances pour visu
        self.neighbors = np.array(self.neighbors)
        self.neighbors = self.neighbors[:, 0, :]
        self.distances = np.array(self.distances)
        self.distances = self.distances[:, 0, :]
        
        if self.verbose:
            print('Prediction done')
        return predicted_genres
    

# Redéfinition de mono_label pour visu_acc_knn
def mono_label(Y_real, Y_pred):
    label_real = Y_real.argmax(axis=1)
    label_pred = Y_pred.argmax(axis=1)
    return (label_real == label_pred).mean()


def visu_acc_knn(list_k, train_features, train_genres, test_features, test_genres):
    """ Fonction qui renvoie un graphe de l'accuracy d'un k-NN,avec k qui varie
    entraîné sur train_features, train_genres,
    et testé sur test_features, test_genres"""
    accuracies = [1, 1]
    for k in tqdm(list_k):
        knn = KNN(k)
        knn.fit(test_features, test_genres)
        pred = knn.predict(test_features, test_genres)
        accuracies.append(mono_label(test_genres, pred))
    plt.plot(list_k, [0.14 for k in list_k], '--', label='Accuracy du hasard')
    plt.plot(list_k, accuracies, label='Accuracy des k-NN')
    plt.title('Accuracies des kNN sur posters bruts')
    plt.legend(loc='best')
    plt.axis([min(list_k), max(list_k), 0, 1.5])
    plt.show()

