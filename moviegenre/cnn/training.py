# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from cnn.model import create_model
from utils import accuracy, display



def train_model(
        Xtr_path, Ytr_path,
        Xtest_path, Ytest_path,
        nb_epochs=1,
        model_path=None, save_path=None,
        verbose=True
):

    Xtr = np.load(Xtr_path)
    Ytr = np.load(Ytr_path)

    Xtest = np.load(Xtest_path)
    Ytest = np.load(Ytest_path)

    if model_path is None :
        model = create_model()
        model.fit(Xtr, Ytr, batch_size=16, epochs=nb_epochs, verbose=verbose, validation_split=0.1)
    else:
        model = load_model(model_path)

    if save_path is not None:
        model.save(save_path)

    Ypred = model.predict(Xtest)

    print("Accuracy on testing set:", accuracy.multi_label(Ytest, Ypred))
    
    return Ypred
    #if verbose:
        #display.plot_test_results(Xtest, Ytest, Ypred, 30, 5, 3)
        #plt.show()
