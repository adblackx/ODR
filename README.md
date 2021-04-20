# <h1 align="center">Ocular Disease Recognition</h1>
<p align="center" style="font-style:italic;">
    Alan Adamiak, Antoine Barbannaud, Sara Droussi, Maya Gawinowski, Ramdane Mouloua, Romain Mussard
</p>

Ceci est le repo GitHub du projet de reconnaissance de maladies oculaires.
<p align="center">
    <img src="./visualisation/rapport/presentation_images/eye_diseases_grid.jpg" style="width:400px;">
</p>

Le dataset utilisé proviens de **Kaggle** vous spouvez le retrouver [ici](https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k).

L'architecture du projet est tirée du [Pytorch Project Template](https://github.com/victoresque/pytorch-template), et nous avons également tiré de nombreuses fonctions des librairies PyTorch et scikit-learn. Nous enregistrons le meilleur modèle en fonction de la valid accuracy, vous retrouverez nos résultats dans des CSV et les `config.json` utilisés dans le dossier `visualisation/rapport/csv`.

---
## Installation des packages

Pour avoir tous les package utilisés ainsi que les versions correspondantes, utilisez:
`pip3 install -r requirements.txt`.

---
## Preparation
Pour pouvoir exécuter le programme qui se trouve dans le main, il faut d'abord appliquer un preprocessing. Le preprocessing se trouve dans le dossier utils (si bug sur windows, déplacer a la racine). Il faut d'abord décompresser l'archive du projet et le mettre dans un dossier (par exemple "data"). Dans le fichier *plot.py*  il faut préciser dans le main le maths, du csv, des données bruts.

Des fichiers CSV et de nouvelles images sont alors générées. Les chemins des fichiers CSV générés et des images sont alors à reporter dans les paramètre `data_dir` et `image_dir` de *config.json* respectivement.

Pour ne pas avoir à modifier manuellement les chemins de lecture et d'écriture nous vous recommandons de décompresser le dataset en faisant en sorte d'avoir les images d'entraînement dans `data/ODIR-5K/ODIR-5K/Training Images` et le fichier xlsx qui décrit les données dans `data/ODIR-5K/ODIR-5K/data.xlsx`.

Toutefois, si le preprocessing ne marche pas vous pouvez télécharger les images pré-traitées [ici](https://mega.nz/file/1lhkDJhQ#mWqVa9TpHKEHM_BTN8EfCWxjL1eFNlxYh9fGUwoRMF4) et les CSV [ici](https://mega.nz/file/UwpmBRyQ#_Ygfeoiw6DksUEi2zlJ8pm1YKQ3MywXuubloDhVyBk0).

## Execution d'une configuration

L'exécution se fait depuis la racine du projet avec la commande `python main.py -c config.json`.

:warning: Deux modèles utilisent une donnée supplémentaire (telle que le sexe ou l'âge): **mymodel** et **myAlexnet**. Il faut alors mettre le paramètre `extended` à true dans le *config.json* pour utiliser les bons Data et Trainer.

:warning: La variable `validation_split` ne correspond pas toujours au véritable découpage du set de validation si `equal_dist == True`. En effet pour éviter d'avoir plus d'images de certaines classes dans notre set de validation que dans le set d'entraînement, le dataloader limite la proportion d'images de chaque classe à 1/3 maximum. La véritable `validation_split` est cependant affichée dans le terminal. Actuellement `validation_split = 0.15` donne un split de 10%. Il aurait été trivial de modifier le dataloader pour que le split donnée soit respecté, cependant nous avons choisi de laisser cette imprécision pour que les résultats présentés dans notre rapport soient reproductibles. La correction sera certainement apportée entre temps sur le repo.

Le champs `dataAug` permet d'activer ou non la data augmentation. `equal_dist` permet quand à lui d'avoir la répartition  des classes la plus équitable possible dans la validation. Les autres paramètres sont les mêmes que pour les autres procédures classiques de CN (batch etc.). Une fois cette étape terminée, vous pouvez exécuter le programme.

## Tensor board
`tensorboard --logdir saved/log/`

`python main.py -c config.json`

`python plot.py -c config_plot.json`

Dans config.js il faut mettre tensorboard à `true`. Puis, pour suivre l'évolution vous devez taper dans un autre terminal en parallèle:

`tensorboard --logdir saved/log/`

On peut faire la même chose mais en affichant les CSV grâce à la classe *plot.py*: il suffit de reporter dans le fichier *config_plot.json* les valeurs pour `data_loader` et `model`. Ensuite, il faut reporter le `model_path` et `affiche` (qui dépend de config.js). Enfin, en se plaçant dans le dossier visualisation on peut afficher en exécutant la commande suivante `python plot.py -c config_plot.json`.

On peut alors afficher trois figures: le loss en fonction de l'époque, l'accuracy en fonctrion de l'époque ou encore la matrice de confusion.

:warning: Attention, `metrics` est ici généré une fois le modèle chargé. Il faut bien faire correspondre *config_plot.json* et le *config.json* enregistré.
