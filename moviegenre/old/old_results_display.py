# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap


def show_img(dataset, posters, labels, ids, index):
    """Shows the image with id at index position in ids"""
    title = dataset.at[ids[index], 'title']
    genre = dataset.at[ids[index], 'genres'][0]+', '+str(ids[index])
    plt.imshow(posters[index])
    plt.title('{} \n {}'.format(title, genre))
    plt.show()


def plot_image(img, true_label, class_names, predictions_array):
    ground_truth = [class_names[k] for k in np.nonzero(true_label)[0]]
    legend = ""
    predicted_label = []
    for k in np.argsort(predictions_array)[-3:][::-1]:
        predicted_label.append(class_names[k])
        legend += class_names[k] + ' {:2.0f}%, '.format(100*predictions_array[k])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    if predicted_label[0] in ground_truth:
        color = 'blue'
    else:
        color = 'red'
    legend += '('+', '.join(ground_truth)+')'
    plt.xlabel('\n'.join(wrap(legend, 30)), color=color, wrap=True)


def plot_value_array(true_label, predictions_array):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(17), predictions_array, color="#777777")
    plt.ylim([0, 1])
    ground_truth = np.nonzero(true_label)[0]
    predicted_labels = np.argsort(predictions_array)[-3:][::-1]
    for k in range(len(ground_truth)):
        thisplot[predicted_labels[k]].set_color('red')
    for k in ground_truth:
        thisplot[k].set_color('blue')


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
