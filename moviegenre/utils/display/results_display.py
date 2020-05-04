# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib import cm 


def show_img(dataset, posters, labels, ids, index):
    """Shows the image with id at index position in ids"""
    title = dataset.at[ids[index], 'title']
    genre = dataset.at[ids[index], 'genres'][0]+', '+str(ids[index])
    plt.imshow(posters[index])
    plt.title('{} \n {}'.format(title, genre))
    plt.show()


def plot_image(img, ground_truth_array, class_names, predictions_array, in_test=True, distance=0):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    true_label = np.argmax(ground_truth_array)
    predicted_label = np.argmax(predictions_array)
    if in_test:
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("Préd:{} {:2.0f}% \n({})".format(class_names[predicted_label],
            100*np.max(predictions_array),
            'L:'+class_names[true_label]),
            color=color)
    else:
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'black'
        plt.xlabel("{} \n Distance:{}".format(class_names[true_label], distance), color=color)

def plot_value_array(ground_truth_array, predictions_array):
    plt.grid(False)
    plt.xticks(range(len(predictions_array)))
    plt.yticks([])

    thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    thisplot[np.argmax(predictions_array)].set_color('red')
    thisplot[np.argmax(ground_truth_array)].set_color('blue')


def plot_test_results(test_posters, test_genres, class_names, predicted_genres, starting_index, num_rows, num_cols):
    num_images = num_rows * num_cols
    num_images = min(num_images, len(test_posters)-starting_index)
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(starting_index, starting_index+num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * (i-starting_index) + 1)
        plot_image(test_posters[i], test_genres[i], class_names, predicted_genres[i])

        plt.subplot(num_rows, 2 * num_cols, 2 * (i-starting_index) + 2)
        plot_value_array(test_genres[i], predicted_genres[i])

    plt.tight_layout()


def plot_neighbors(test_posters, test_genres, class_names, predicted_genres, starting_index, num_images, train_posters, train_genres, neighbors, distances, method_for_title):
    """Affiche num_images posters et leurs plus proches voisins.
    Les posters sont choisis à partir de l'index starting_index de test_posters
    neighbors est un vecteur : neighbors[i] contient les indexes dans
    training_set des plus proches voisins   de test_posters[i].
    distances contient les distances des plus proches voisins, dans le même
    format que neighbors.
    test_genres est les labels des éléments du testing_set.
    predicted_genres est les genres prédits du testing_set par le knn.
    """
    k = len(neighbors[0])
    num_images = min(num_images, len(test_posters)-starting_index)
    plt.figure(figsize=(2 * (k+1), 2 * num_images))
    plt.title('Plus proches voisins après '+method_for_title)
    
    ind = 0
    for i in range(starting_index, starting_index+num_images):
        ind += 1
        plt.subplot(num_images, k+1, ind)
        # Affichage du poster test
        plot_image(test_posters[i], test_genres[i], class_names, predicted_genres[i])
        # Affichage de ses plus proches voisins
        for n_neighbor in range(k):
            ind += 1
            plt.subplot(num_images, k+1, ind)
            plot_image(train_posters[neighbors[i, n_neighbor]], train_genres[neighbors[i, n_neighbor]], class_names, test_genres[i], in_test=False, distance=int(distances[i, n_neighbor]))
    plt.tight_layout()


# def histogram(ground_truth_arrays, predicted_arrays, genre_index, title):
#     predicted_labels = np.argmax(predictions_arrays, axis=1)
#     true_labels = np.argmax(ground_truth_arrays, axis=1)
#     to_keep = np.argwhere(true_labels == genre_index)
#     predicted_labels = predicted_labels[to_keep]
#     true_labels = true_labels[to_keep]


def histogram(test_genres, predicted_genres, kneighbors, genres, save=False):
    genres_inv = {genres[k]: k for k in genres.keys()}
    predictions = np.array([genres_inv[k] for k in np.argmax(predicted_genres, axis=1)])
    ground_truth = np.array([genres_inv[k] for k in np.argmax(test_genres, axis=1)])

    # results_per_genre : matrice donc les lignes sont les vrais genres, et les colonnes sont les genres prédits
    results_per_genre = {
        genre_true : {genre_pred : 0 for genre_pred in genres}
        for genre_true in genres
    }
    #print("results per genre", results_per_genre)
    #print('results per genre ligne', results_per_genre['Action'].values())
    # total_per_genre : vecteur qui comptabilise le nombre de représentants de chaque genre
    total_per_genre = {
        genre : 0
        for genre in genres
    }
    # Mise à jour de results_per_genre et total_per_genre en fonction des prédictions
    n = len(predictions)
    for i in range(n):
        results_per_genre[ground_truth[i]][predictions[i]] += 1
        total_per_genre[ground_truth[i]] += 1

    # Accuracy
    accuracy = 0
    for genre in genres:
        accuracy += results_per_genre[genre][genre]
    accuracy /= len(test_genres)

    # Visualisation:
    genres_list = list(genres)
    for iterateur in genres_list:
        plt.figure(figsize=(15, 5))
        plt.title('Prédictions sur les films de genre ' + iterateur + ' kneighbors='+str(kneighbors) +", Accuracy totale:" + str(accuracy))
        plt.bar(genres_list, results_per_genre[iterateur].values())
        plt.show()
        if save:
            plt.savefig('../results/Resnet+kNN/'+iterateur+'_k='+str(kneighbors)+'.png')


def ConfusionMatrix_display(test_genres, predicted_genres, genres, method_for_title):
    """ Returns and displays the confusion matrix between test_genres and 
    predicted_genres. 
    genres is the dictionnary of genres used. 
    method_for_title is a string whhich explains the classifier used to build
    predicted_genres.    
    """
    genres_list = list(genres)
    print(genres_list)
    # COnvert predictions to strings
    genres_inv = {genres[k]: k for k in genres.keys()}
    predictions = np.array([genres_inv[k] for k in np.argmax(predicted_genres, axis=1)])
    ground_truth = np.array([genres_inv[k] for k in np.argmax(test_genres, axis=1)])
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=genres_list, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=genres_list)
    
    disp.plot(cmap=cm.coolwarm_r, xticks_rotation='vertical')
    plt.title('Matrice de confusion, '+ method_for_title)
    return(conf_matrix)


