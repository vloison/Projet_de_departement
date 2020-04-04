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
    

def plot_image(i, img, true_label, class_names, predictions_array):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == np.argmax(true_label):
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[np.argmax(true_label)]
        ),
        color=color
    )


def plot_value_array(i, true_label, predictions_array):
    predictions_array, true_label = predictions_array[i], true_label[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(17), predictions_array, color="#777777")
    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[np.argmax(true_label)].set_color('blue')


def plot_test_results(Xtest, Ytest, class_names, predictions, starting_index, num_rows, num_cols):
    num_images = num_rows * num_cols

    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i + starting_index, Xtest, Ytest, class_names, predictions)

        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i + starting_index, Ytest, predictions)

    plt.tight_layout()
