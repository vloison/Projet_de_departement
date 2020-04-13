## TO DO : pour le 23/03/2020

- Forme finale de la BDD, training_set et testing_set finaux, passage en monolabel -> Maxime
- Checker ResNet -> Alex
- Features (Histogramme RGB, Histogramme LAB) : Nico
- k-NN, checker la distance de transport optimal -> Vivi

### Guidelines

- Se concentrer sur les genres prédominants
- L'accuracy seule d'une méthode ne veut rien dire, il faut la comparer à l'accuracy d'autres méthodes.
- Sauvegarder les visualisations au fur et à mesure.
- Faire du monolabel au moins jusqu'à ce qu'on ait des résultats satisfaisants.
- Privilégier le fine tuning à l'entraînement from scratch.


##Pour plus tard :
- Regarder ce qui se fait sur Google Cloud Platform : Utilisation boîte noire
- Interprétabilité du résultat : éventuellement revisualiser les infos processées par le modèle. Auto-encodeurs ?
- Couche linéaire + Softmax & Crossentropy -> Modèle linéaire interprétable.
- Donner des représentations de ResNet en argument à un k-NN
- Ajouter une couche de classifieur à la fin d'un ResNet déjà entraîné et entraîner uniquement le classifieur
