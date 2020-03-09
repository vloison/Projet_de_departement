from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import imageio
from preprocessing import show_img, preprocess, MOVIES
import matplotlib.pyplot as plt
from tqdm import tqdm

# CONSTANTS
k = 3  # k of the k-NN : number of closest neigbors
ind = 608  # indice of the poster to be treated in the original database
training_size = 1000  # Size of the training set
testing_size = 100   # Size of the testing set


# LOAD DATA
fileend  = '_tr='+str(training_size)+'_test='+str(testing_size)+'.npy'
Xtr = np.load('./sets/Xtr'+fileend)
print("Shape of Xtr:", Xtr.shape)
Xtrprim = np.reshape(Xtr, (len(Xtr), 150*100*3))
print("Shape of Xtrprim:", Xtrprim.shape)
Ytr = np.load('./sets/Ytr'+fileend)
print("Shape of Ytr", Ytr.shape)
training_set = np.load('./sets/training_set'+fileend)
testing_set = np.load('./sets/testing_set'+fileend)


# KNN FUNCTION AND RESULTS

def KNN(Xtr, Xtrprim, Ytr, training_set, ind, k, print_results=False):
    """ Calculates the genre of movie of indice ind using a k-NN approach.
    Xtr is the set of posters among which the closest neighbors will be found.
    Ytr stocks the genra of each poster of Xtr.
    training_set stocks the indice of each poster of Xtr in the original
    database. """

    # Create k-NN operator
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtrprim, Ytr)
    # Preprocess the poster to be classified
    path = Path('../posters/' + str(ind) + '.jpg')
    x = preprocess(imageio.imread(path), (150, 100, 3))
    if print_results:
        plt.imshow(x)
        plt.show()
    # Classification
    x = np.reshape(x, (150*100*3))
    genre_ind = np.where(neigh.predict([x])[0] == 1)
    genre = MOVIES.genre_1.unique()[genre_ind]
    # Print results
    if print_results:
        print("Title of the movie:", MOVIES.at[ind, 'title'],
              ", Label: ", MOVIES.at[ind, 'genre_1'])
        print("Prediction for poster : ", genre)
        # DIsplay neighbors
        neighbors = neigh.kneighbors([x], return_distance=False)
        for neighbor in neighbors[0]:
            show_img(MOVIES, Xtr, Ytr, training_set, neighbor)
    return genre


#KNN(Xtr, Xtrprim, Ytr, training_set, ind, k, print_results=True)


def test_KNN(Xtr, Xtrprim, Ytr, training_set, testing_set, k):
    error = 0
    for i in tqdm(testing_set):
        res = KNN(Xtr, Xtrprim, Ytr, training_set, i, k)
        if res.size == 0 or res != MOVIES.at[i, 'genre_1']:
            error += 1
    error = error/len(testing_set)
    return(error)


print(test_KNN(Xtr, Xtrprim, Ytr, training_set, testing_set, k))
        