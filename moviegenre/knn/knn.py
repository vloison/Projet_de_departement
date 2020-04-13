import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# KNN FUNCTION AND RESULTS

def KNN(dataset, Xtr, Xtrprim, Ytr, training_set, Xtest, testing_set, ind, k, image_size, genres_dict, print_results=False):
    """ Calculates the genre of movie of indice ind in the testing set using a k-NN approach.
    Xtr is the set of posters among which the closest neighbors will be found.
    Ytr stocks the genra of each poster of Xtr.
    training_set stocks the indice of each poster of Xtr in the original
    database. """

    # Create k-NN operator
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtrprim, Ytr)
    # Preprocess the poster to be classified
    x = Xtest[ind]
    if print_results:
        print("Movie to be labellised:")
        plt.imshow(x)
        plt.show()
    # Classification
    x = np.reshape(x, (image_size[0]*image_size[1]*image_size[2]))
    
    prediction = neigh.predict([x])
    print('Prediction:', prediction)
    #genre_ind = np.where(neigh.predict([x])[0] == 1)[0]
    #print('genre_ind', genre_ind)
    #genre = genres_dict[list(genre_ind)]
    # Print results
    if print_results:
        titre = dataset.loc[dataset['allocine_id'] == testing_set[ind], ['title']].values[0]
        print("Title of the movie:", titre)
        print("Label: ", dataset.loc[dataset['allocine_id'] == testing_set[ind], ['genres']].values[0])
        #print("Prediction for poster : ", genre_ind)
        # DIsplay neighbors
        neighbors = neigh.kneighbors([x], return_distance=False)
        print("Closest neighbors:")
        for neighbor in neighbors[0]:
            plt.imshow(Xtr[neighbor])
            plt.show()
            titre = dataset.loc[dataset['allocine_id'] == training_set[neighbor], ['title']].values[0]
            label = dataset.loc[dataset['allocine_id'] == training_set[neighbor], ['genres']].values[0]
            print("Title of the movie:", titre)
            print("Label: ", label)
    return prediction


def test_KNN(dataset, Xtr, Xtrprim, Ytr, training_set, Xtest, Ytest, testing_set, k, accuracy_funct, image_size, genres_dict):
    # Initialize k-NN parameters
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtrprim, Ytr)
    
    accuracy = 0
    for i in tqdm(range(len(testing_set))):
        # Compute x
        x = Xtest[i]
        x = np.reshape(x, (image_size[0]*image_size[1]*image_size[2]))
        # Prediction
        prediction = neigh.predict([x])
        # Compute accuracy
        accuracy += accuracy_funct(Ytest[i], prediction)
    accuracy = accuracy/len(testing_set)
    return(accuracy)
        