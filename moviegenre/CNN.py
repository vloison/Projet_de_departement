import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.metrics import Accuracy

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(150, 100, 3)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(33, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer=Adagrad(), metrics=["accuracy"])

Xtr = np.load("../data/sets/Xtr_tr=2000_test=100.npy")
Ytr = np.load("../data/sets/Ytr_tr=2000_test=100.npy")

Xtest = np.load("../data/sets/Xtest_tr=2000_test=100.npy")
Ytest = np.load("../data/sets/Ytest_tr=2000_test=100.npy")

model.fit(Xtr, Ytr, batch_size=16, epochs=5, verbose=1, validation_split=0.1)

model.evaluate(Xtest, Ytest, batch_size=1, verbose=1)
