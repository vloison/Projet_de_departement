from pathlib import Path
from CNN.training import train_model
import pandas as pd

Xtr_path = Path("../data/sets/Xtr_tr=2000_test=100.npy")
Ytr_path = Path("../data/sets/Ytr_tr=2000_test=100.npy")

Xtest_path = Path("../data/sets/Xtest_tr=2000_test=100.npy")
Ytest_path = Path("../data/sets/Ytest_tr=2000_test=100.npy")

model_path = Path("../data/first_model.h5")

MOVIES = pd.read_csv("../data/clean_poster_data.csv", index_col=0)
class_names = MOVIES.genre_1.unique()

train_model(class_names, Xtr_path, Ytr_path, Xtest_path, Ytest_path, model_path)
