import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import show_img
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from tensorflow.keras.optimizers import Adagrad
# from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import load_model

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(150, 100, 3)))
# model.add(Conv2D(64, (3, 3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# 
# model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
# model.add(Conv2D(64, (3, 3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# 
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(17, activation="sigmoid"))
# 
# model.compile(loss="binary_crossentropy", optimizer=Adagrad(), metrics=["accuracy"])

model = load_model('../data/first_model.h5')
Xtr = np.load("../data/sets/Xtr_tr=2000_test=100.npy")
Ytr = np.load("../data/sets/Ytr_tr=2000_test=100.npy")
IDStr = np.load("../data/sets/IDStr_tr=2000_test=100.npy")

Xtest = np.load("../data/sets/Xtest_tr=2000_test=100.npy")
Ytest = np.load("../data/sets/Ytest_tr=2000_test=100.npy")
IDStest = np.load("../data/sets/IDStest_tr=2000_test=100.npy")

MOVIES = pd.read_csv("../data/clean_poster_data.csv")
class_names = MOVIES.genre_1.unique()
# model.fit(Xtr, Ytr, batch_size=16, epochs=5, verbose=1, validation_split=0.1)

# model.evaluate(Xtest, Ytest, batch_size=1, verbose=1)
predictions = model.predict(Xtest)
print(predictions[0])
# model.save('../data/first_model.h5')

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

starting_index = 30
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i+starting_index, predictions[i+starting_index], Ytest, Xtest)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i+starting_index, predictions[i+starting_index], Ytest)
plt.tight_layout()
plt.show()

