## TO DO : pour le 27/04/2020

- Mtrices de confusion, F-score, intégration du k-NN dans la pipeline -> Maxime
- Accès aux différences couches de ResNet, PCA -> Alex
- Peaufiner les histograppes (distance, bins), et regarder les résultats du knn dessus : Nico
- Intégrer le multiclasses au kNN, checker les résultats sur des images plus grandes -> Vivi

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
