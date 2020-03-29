import numpy as np
from sklearn.neighbors import KNeighborsClassifier

SEED = 7
np.random.seed(SEED)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == np.argmax(true_label):
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[np.argmax(true_label)]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(17), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[np.argmax(true_label)].set_color('blue')


class MLPipeline:

    def __init__(self, params, verbose=False):
        print('Initialising Modelling Pipeline...')
        self.verbose = verbose
        self.MODELS = {'KNN': KNeighborsClassifier(n_neighbors=params['n_neighbors'],
                        'CNN':}

    def ingestData(self, train_data, test_data, target_var):
        print('\n################## INGESTION #############')
        if self.verbose: print('Ingesting Data into Pipeline...')
        self.train_data = train_data
        self.test_data = test_data
        self.target_var = target_var
        print('###########################################\n\n')

    def cleanTransformData(self):
        pass

    def fitModel(self):
        pass

    def evaluateModel(self, metrics, saved_model_path = None):
        pass

    def calcMetrics(self):
        pass
