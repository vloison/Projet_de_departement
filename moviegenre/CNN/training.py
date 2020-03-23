import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from CNN.model import create_model
from utils import accuracy, display


MOVIES = pd.read_csv("../data/clean_poster_data.csv", index_col=0)
class_names = MOVIES.genre_1.unique()


def train_model(Xtr_path, Ytr_path, Xtest_path, Ytest_path, model_path=None, save_path=None):
    if model_path is None :
        model = create_model()
    else:
        model = load_model(model_path)

    Xtr = np.load(Xtr_path)
    Ytr = np.load(Ytest_path)

    Xtest = np.load(Xtest_path)
    Ytest = np.load(Ytest_path)

    model.fit(Xtr, Ytr, batch_size=16, epochs=5, verbose=1, validation_split=0.1)

    if save_path is not None:
        model.save(save_path)

    Ypred = model.predict(Xtest)

    print("Accuracy on testing set:", accuracy.mono_label(Ytest, Ypred))

    display.plot_test_results(Xtest, Ytest, Ypred, 30, 5, 3)
    plt.show()
