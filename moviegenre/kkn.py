from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import imageio
from preprocessing import show_img, preprocess, MOVIES
import matplotlib.pyplot as plt

X = np.load('../data/all_clean_posters.npy')
Y = np.load('../data/all_clean_genres.npy')
IDS = np.load('../data/all_clean_ids.npy')

print(IDS.dtype)
X = np.reshape(X, (len(X), 150*100*3))
print(np.shape(X))
print(np.shape(Y))
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,Y)
path = Path('../data/posters/1046.jpg')
x = preprocess(imageio.imread(path), (150, 100, 3))
plt.imshow(x)
plt.show()
x = np.reshape(x, (150*100*3))

print(neigh.predict([x]))
neighbors = neigh.kneighbors([x], return_distance=False)
print(neighbors)
print(MOVIES.genre_1.unique())
print(MOVIES.at[1046, 'title'], MOVIES.at[1046, 'genre_1'])

for neighbor in neighbors[0]:
    show_img(MOVIES, X, Y, IDS, neighbor)
