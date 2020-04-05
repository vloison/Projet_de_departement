import argparse
from pathlib import Path
import pandas as pd
from cnn.training import train_model


parser = argparse.ArgumentParser()
parser.add_argument("--save", help="save model", action="store_true")
parser.add_argument("--load", help="load model", action="store_true")
args = parser.parse_args()

Xtr_path = Path("../data/sets/Xtr_tr=2000_test=100.npy")
Ytr_path = Path("../data/sets/Ytr_tr=2000_test=100.npy")

Xtest_path = Path("../data/sets/Xtest_tr=2000_test=100.npy")
Ytest_path = Path("../data/sets/Ytest_tr=2000_test=100.npy")

if args.load:
    train_model(Xtr_path, Ytr_path, Xtest_path, Ytest_path, nb_epochs=5, model_path='../data/first_model.h5')
elif args.save:
    train_model(Xtr_path, Ytr_path, Xtest_path, Ytest_path, nb_epochs=5, save_path='../data/first_model.h5')
else:
    train_model(Xtr_path, Ytr_path, Xtest_path, Ytest_path, nb_epochs=5)

