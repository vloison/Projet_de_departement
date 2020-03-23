import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from CNN.model import create_model
from utils import accuracy, display


MOVIES = pd.read_csv("../data/clean_poster_data.csv", index_col=0)
class_names = MOVIES.genre_1.unique()


def train_model(
        class_names,
        Xtr_path, Ytr_path,
        Xtest_path, Ytest_path,
        nb_epochs=1,
        model_path=None, save_path=None,
        verbose=True
):
    if model_path is None :
        model = create_model()
    else:
        model = load_model(model_path)

    Xtr = np.load(Xtr_path)
    Ytr = np.load(Ytr_path)

    Xtest = np.load(Xtest_path)
    Ytest = np.load(Ytest_path)

    model.fit(Xtr, Ytr, batch_size=16, epochs=nb_epochs, verbose=verbose, validation_split=0.1)

    if save_path is not None:
        model.save(save_path)

    Ypred = model.predict(Xtest)

    print("Accuracy on testing set:", accuracy.mono_label(Ytest, Ypred))

    if verbose:
        display.plot_test_results(Xtest, Ytest, class_names, Ypred, 30, 5, 3)
        plt.show()
