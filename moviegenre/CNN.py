from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.models import load_model


def CNN(shape, nb_genres):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=shape))
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
    model.add(Dense(nb_genres, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=Adagrad(), metrics=["accuracy"])

    return model

model = load_model('../data/first_model.h5')
Xtr = np.load("../data/sets/Xtr_tr=2000_test=100.npy")
Ytr = np.load("../data/sets/Ytr_tr=2000_test=100.npy")
IDStr = np.load("../data/sets/IDStr_tr=2000_test=100.npy")

Xtest = np.load("../data/sets/Xtest_tr=2000_test=100.npy")
Ytest = np.load("../data/sets/Ytest_tr=2000_test=100.npy")
IDStest = np.load("../data/sets/IDStest_tr=2000_test=100.npy")

MOVIES = pd.read_csv("../data/clean_poster_data.csv", index_col=0)
class_names = MOVIES.genre_1.unique()
# model.fit(Xtr, Ytr, batch_size=16, epochs=5, verbose=1, validation_split=0.1)

# model.evaluate(Xtest, Ytest, batch_size=1, verbose=1)
predictions = model.predict(Xtest)
print(predictions[0])
# model.save('../data/first_model.h5')


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

