from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import imageio
from preprocessing import show_img, preprocess, MOVIES
import matplotlib.pyplot as plt

# CONSTANTS
k = 3  # k of the k-NN : number of closest neigbors
ind = 405 # indice of the poster to be treated in the original database
training_size = 1000  # Size of the training set
testing_size = 30   # Size of the testing set


# LOAD DATA

X = np.load('../data/all_clean_posters.npy')
print("Size of X:", X.shape)
Y = np.load('../data/all_clean_genres.npy')

IDS = np.load('../data/all_clean_ids.npy')
print("Size of IDS:", np.max(IDS))
print("IDS", IDS)
# CONVERT DATA FOR K-NN COMPUTE
#print("Type of the coordinates of IDS: " + str(IDS.dtype))
Xprim = np.reshape(X, (len(X), 150*100*3))
# print("New shape of X" + str(np.shape(X)))
print("Shape of Y" + str(np.shape(Y)))


# DEFINITION OF THE TRAINING AND TESTING SETS
def sets(dataset, IDS, training_size, testing_size):
    """ Builds a training_set and a testing_set as np.arrays, lists of
    indices. Each set has the size given in argument """
    IDS2 = [IDS[i] for i in range(len(IDS))]
    np.random.shuffle(IDS2)
    training_set = IDS2[:training_size]
    testing_set = IDS2[training_size:training_size+testing_size]
    return training_set, testing_set


def prepare_training_set(X, Y, training_set, IDS):
    Xtr = np.array([X[np.where(IDS == i)] for i in training_set])
    Xtr = np.reshape(Xtr, (len(Xtr), 150, 100, 3))
    Ytr = np.array([Y[np.where(IDS == i)] for i in training_set])
    Ytr = np.reshape(Ytr, (len(Ytr), 33))
    return Xtr, Ytr


training_set, testing_set = sets(MOVIES, IDS, training_size, testing_size)
Xtr, Ytr = prepare_training_set(X, Y, training_set, IDS)

# KNN FUNCTION AND RESULTS

print(IDS)
def KNN(Xtr, Ytr, training_set, ind, k ):
    # Create k-NN operator
    Xtrprim = np.reshape(Xtr, (len(Xtr), 150*100*3))
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtrprim, Ytr)
    # Preprocess the poster to be classified
    path = Path('../posters/' + str(ind) + '.jpg')
    x = preprocess(imageio.imread(path), (150, 100, 3))
    plt.imshow(x)
    plt.show()
    # Resize the poster for classification
    
    x = np.reshape(x, (150*100*3))
    # Print results
    genre_ind = np.where(neigh.predict([x])[0] ==1)
    genre = MOVIES.genre_1.unique()[genre_ind]
    print("Prediction for poster : ", genre)
    
    neighbors= neigh.kneighbors([x], return_distance=False)

    print("Neighbors", neighbors)

    # RETURN THE RESULTS
    print("All genres:", MOVIES.genre_1.unique())
    print("Title of the movie:", MOVIES.at[ind, 'title'],
          ", Label: ", MOVIES.at[ind, 'genre_1'])
    for neighbor in neighbors[0]:
        show_img(MOVIES, Xtr, Ytr, training_set, neighbor)
    print(genre)
    return genre

KNN(Xtr, Ytr, training_set, ind, k)