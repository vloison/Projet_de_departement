# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def show_img(dataset, posters, labels, ids, index):
    """Shows the image with id at index position in ids"""
    title = dataset.at[ids[index], 'title']
    genre = dataset.at[ids[index], 'genres'][0]+', '+str(ids[index])
    plt.imshow(posters[index])
    plt.title('{} \n {}'.format(title, genre))
    plt.show()


def plot_image(img, ground_truth_array, class_names, predictions_array, in_test=True):
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

        plt.xlabel("{} {:2.0f}% \n({})".format(class_names[predicted_label],
            100*np.max(predictions_array),
            class_names[true_label]),
            color=color) 
    else:
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'black'
        plt.xlabel(class_names[true_label], color=color)

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


def plot_neighbors(test_posters, test_genres, class_names, predicted_genres, starting_index, num_images, train_posters, train_genres, neighbors):
    
    k = len(neighbors[0])
    num_images = min(num_images, len(test_posters)-starting_index)
    plt.figure(figsize=(2 * (k+1), 2 * num_images))
    ind = 0
    for i in range(starting_index, starting_index+num_images):
        ind += 1
        plt.subplot(num_images, k+1, ind)
        plot_image(test_posters[i], test_genres[i], class_names, predicted_genres[i])
        for n_neighbor in range(k):
            ind += 1
            plt.subplot(num_images, k+1, ind)
            plot_image(train_posters[neighbors[i, n_neighbor]], train_genres[neighbors[i, n_neighbor]], class_names, test_genres[i], in_test=False)

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
    genres_list = list(config['genres'])
    for iterateur in genres_list:
        plt.figure(figsize=(15, 5))
        plt.title('Prédictions sur les films de genre ' + iterateur + ' k='+str(k) +", Accuracy totale:" + str(accuracy))
        plt.bar(genres_list, results_per_genre[iterateur].values())
        plt.show()
        if save:
            plt.savefig('../results/Resnet+kNN/'+iterateur+'_k='+str(kneighbors)+'.png')