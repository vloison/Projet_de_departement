from pathlib import Path
import pandas as pd
from cnn.training import train_model


Xtr_path = Path("../data/sets/Xtr_tr=2000_test=100.npy")
Ytr_path = Path("../data/sets/Ytr_tr=2000_test=100.npy")

Xtest_path = Path("../data/sets/Xtest_tr=2000_test=100.npy")
Ytest_path = Path("../data/sets/Ytest_tr=2000_test=100.npy")

model_path = Path("../data/first_model.h5")

train_model(Xtr_path, Ytr_path, Xtest_path, Ytest_path, nb_epochs=1, save_path='../data/first_model.h5')
