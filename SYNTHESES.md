## Synthèse Maxime
Lien du github: https://github.com/davideiacobs/-Movie-Genres-Classification-from-their-Poster-Image-using-CNNs.

À pour but de faire un modèle prédisant les genres d'un film à partir de son poster (comme nous, aussi dans le cadre multi-label).
Se sert de la base de données IMDB en récupérant les infos directement depuis les pages webs: plus compliqué que dans notre cas, demande plus de nettoyage de bdd aussi.

J'ai repris dans les grandes lignes ce qui a été fait pour les scripts 'database.py' et 'preprocessing.py', en adaptant pour notre base de données.

Le notebook se divise en 5 étapes:
- La première est le *webscrapping*: pas besoin d'en faire pour nous.
- La deuxième est le téléchargement des posters: cf 'database.py'
- La troisième est la manipulation de la base de données. Il enlève les films classés en tant que "TV Movie" et "Foreign". Il ne conserve plus que 8500 films environ pour que se soit raisonnable à traiter. Mais surtout il procède à une *_oversampling approach_*: dans l'échantillon sélectionné, on ajoute plus d'occurences de genres peu représentés. Si on sélectionnait tous les films au hasard, on se retrouverait en très grande majorité avec des drames et des comédies, ce qui ne permet pas au modèle de bien apprendre pour les autres genres. Ceci n'est pas implémenté dans les deux scripts python actuellement.
- La quatrième étape est le pré-traitement des images et la construction de la base de données finale: cf 'preprocessing.py'.
- La cinquième étape est la construction et l'implémentation du CNN. Il utilise keras (aujourd'hui plutôt utilisé tf.keras que keras). Les quatre premières couches sont des *Conv2D layers*, puis il y a un *Flatten layer* utilisé pour faire la connection avec les deux *Dense layers* qui suivent. L'activation pour la dernière couche est une sigmoïde car on a un problème multi-label (*The activation for the last layer is sigmoid since we are dealing with a multi-label classication problem.*).

En faisant une phase de *training* sur 4000 films, il obtient une loss (*binary cross-entropy*) de 0.39 à la fin. En faisant une prédiction sur 100 instances de test, il obtient 64% de succès.
