## Présentation du contexte
2 personnes dans l'équipe data de France : Stefano Perasso et Lise Regnier.
Plusieurs projets, dont la mise en place d'un système de dashboard pour visualiser les données et d'outils de data science pour prévision de box office etc...
Si on a besoin de puissance de calcul, Warner peut mettre un serveur google cloud à disposition pour faire tourner les calculs.

## Présentation
l'équipe data de Warner a travaillé sur des algos de prédiction d'audience d'un film (segment d'âge par exemple).
Pour prédire ce genre de chose, connaître le genre du film est important. Actuellement, Warner récupère cette information  à partir d'allociné et de CBO (cbo-boxoffice.com) mais quelquefois l'info est mauvaise.
L'idée est de faire un algorithme indépendant qui prédise le(s) genre(s) d'un film, pour confronter son résultat à celui des sites référents.
C'est donc un *problème de classification multilabel*
En fonction de l'avancée du projet, nous pourrons explorer d'autres utilisations de l'algorithme.

## Base de donnée
Nous disposons de la BDD d'allociné. C'est un fichier excel dont les colonnes sont : titre du film / jusqu'à 3 genres (classés par ordre d'importance décroissant)/ lien vers le poster. Les genres sont attribués manuellement, en se basant sur le synopsis des films.

## Démarche
Nous commençons par une exploration bibliographique. En parallèle, nous faisons des premiers algorithmes simples.
L'objectif est de se familiariser avec la bdd et les outils, et d'avoir un premier résultat auquel nous comparerons nos futurs algorithmes.
Nous commençons par un réseau k-NN monolabel où la distance est calculée par rapport aux pixels des posters.
Nous créons ensuite un réseau de deep learning "force brute".
Nous pourrons ensuite envisager un réseau de deep learning, avec des features bien choisis. 
