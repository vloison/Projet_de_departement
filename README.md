Dans le cadre de notre projet de département IMI, nous avons travaillé avec Warner Bros pour mettre au point un outil de classification d’affiches de films. Le but était de pouvoir prédire les différents genres d’un film à partir de son affiche. Pour ce projet, nous étions accompagnés par Lise Régnier, data scientist chez Warner Bros. Notre rapport final est *rapport.pdf*.


# Installation
* Utiliser Jupyter Notebook avec une machine gcloud: https://medium.com/@kn.maragatham09/installing-jupyter-notebook-on-google-cloud-11979e40cd10
* Utiliser le GPU avec Tensorflow 2.2.0:
    - Installer les drivers NVIDIA: se rendre ici https://www.nvidia.com/Download/index.aspx?lang=en-us, en rentrant son modèle de GPU et CUDA 10.1
    - Version 10.1 de CUDA requise, pas la 10.2 attention. Trouvable ici: https://developer.nvidia.com/cuda-10.1-download-archive-base. Des problèmes d'installation avec le .deb, utiliser le runfile à la place.
    - Installer cuDNN: doit d'abord s'inscrire au programme développeur de NVIDIA, puis récupérer le .deb sur cette page https://developer.nvidia.com/rdp/cudnn-download. Bien prendre celui qui correspond à CUDA 10.1 (libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb). Comme authentification nécessaire, doit d'abord télécharger le fichier en local avant de l'importer sur la machine.
* Une fois pip installé, faire _pip install requirements.txt_ pour installer les modules nécessaires.


# Structure du projet
- _data_: contient les posters, les fichiers .csv et les modèles sauvegardés. Le modèle final est *final_model.h5*.
- _moviegenre_: contient les fichiers python et les notebooks.
    * *transfer_learning.ipynb*: permet l'entraînement du réseau à base de transfer learning, et visualisation des résultats.
    * *resnet_knn.ipynb*: construit le modèle resnet + knn, visualisation des résultats. Vieux preprocessing
    * *features.ipynb*: utilise les histogrammes de couleurs. Sûrement des bugs et vieux preprocessing
    * *exemple_loader.ipynb*: base de code pour utiliser le data loader et faire l'extraction de features. Peut être complété en se basant sur *resnet_knn.ipynb* pour reproduire des résultats similaires mais avec le data loader et une data augmentation possible.
    * *get_data.py*: à exécuter la première fois. Télécharge les films. Décommenter la fin du fichier pour faire le vieux preprocessing. Tout n'est pas à jour ni utile, le laisser comme ça si uniquement utilisation de *transfer_learning.ipynb*
    * *compute_features.py*: calcul les histogrammes de couleur, basé sur *get_data.py*. Ne pas s'en servir.
    * *default_config.py*: fichier de configuration pour *get_data.py*.
    * *cnn*: premier dossier pour l'utilisation des CNN. Pas à jour, ne sert plus.
    * *knn*: implémentation du KNN.
    * *preprocessing*: permet de nettoyer la base de données, de télécharger les posters. Permet aussi de faire le vieux preprocessing.
    * *utils*: contient des fonctions diverses, notamment de visualisation. Tout n'est pas à jour ou utile, certaines fonctions des notebooks pourraient se trouver dans ce dossier.
- _results_: contient les visualisation de nos résultats. Ceux dans _transfer_ sont à jour, pas forcément le cas dans les autres dossiers.
- _ressources_: contient les ressources de départ et notre premier draft.


# Utilisation
Le dossier _moviegenre_ traduit l'évolution du projet et certains fichiers présents ne sont plus si importants. En première utilisation, exécutez le fichier *get_data.py* pour téléchargez les posters. Cela téléchargera tous les posters utilisables dans un même dossier.

Nous n'avions pas utilisé de _data loader_ pendant une bonne partie du projet, et l'utilisation n'est vraiment faite que pour le _transfer learning_ avec une tête de classification.
Ainsi si vous utilisez les notebooks resnet_knn et features (pour les histogrammes de couleur) en l'état, le preprocessing s'effectuera avec ce que nous avons implémenté directement et qui n'est pas du tout optimal.
Le notebook transfer_learning_resnet utilise par contre le _data loader_ mais pour cela il est nécessaire de déplacer les posters de films, ce qui est fait avec le fichier _movefiles.py_ (ou première cellule de ce notebook).

Nous vous conseignons d'utiliser le _data loader_ et de d'abord regarder le notebook transfer_learning_resnet, puis d'adapter les codes des autres notebooks en se basant sur *exemple_loader.ipynb* et de ne pas utiliser le preprocessing de départ.
