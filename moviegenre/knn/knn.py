from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from tqdm import tqdm


class KNN(KNeighborsClassifier):
    def __init__(self, n_neighbors, verbose=True, weights='distance'):
        super().__init__(n_neighbors=n_neighbors, weights=weights)
        self.verbose = verbose
        self.neighbors = None

    def predict(self, test_features, train_genres):
        self.neighbors = []
        generator = tqdm(range(len(test_features))) if self.verbose else range(len(test_features))
        predicted_genres = np.zeros((len(test_features), train_genres.shape[1]))
        if self.verbose:
            print('Predicting...')
        for i in generator:
            neighbors = self.kneighbors([test_features[i]], return_distance=False)
            self.neighbors.append(neighbors)
            label_for_max = train_genres[neighbors]
            label_for_max = np.sum(label_for_max, axis=1)
            ind_genre = np.argmax(label_for_max)
            predicted_genres[i][ind_genre] = 1
        self.neighbors = np.array(self.neighbors)
        self.neighbors = self.neighbors[:,0,:]
        if self.verbose:
            print('Prediction done')
        return predicted_genres    



