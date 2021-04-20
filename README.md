# <h1 align="center">Ocular Disease Recognition</h1>
_Alan Adamiak, Antoine Barbannaud, Sara Droussi, Maya Gawinowski, Ramdane Mouloua, Romain Mussard_

Ceci est le repo GitHub du projet de reconnaissance de maladies oculaires.
<p align="center">
    <img src="./visualisation/rapport/presentation_images/eye_diseases_grid.jpg" style="width:400px;">
</p>

L'architecture du projet est tirée du [Pytorch Project Template](https://github.com/moemen95/Pytorch-Project-Template), et nous avons également tiré de nombreuses fonctions des librairies PyTorch et scikit-learn. Nous enregistrons le meilleur modèle en fonction de la valid accuracy, vous retrouverez nos résultats dans les CSV [ici](https://mega.nz/file/UwpmBRyQ#_Ygfeoiw6DksUEi2zlJ8pm1YKQ3MywXuubloDhVyBk0).

---
## Installation des packages

Pour avoir tous les package utilisés ainsi que les versions correspondantes, utilisez: `pip3 install -r requirements.txt`.

---
## Preparation
Pour pouvoir exécuter le programme qui se trouve dans le main, il faut d'abord appliquer un preprocessing. Le preprocessing se trouve dans le dossier utils ( ou bien à la racine en raison de bugs pour certains sur windows). Il faut d'abord décompresser l'archive du projet, et le mettre dans un dossier par exemple "data", et dans le fichier plot.py dans le main, il faut alors préciser le maths, du csv, des données bruts

Toutefois, si le preprocessing ne marche pas, vous pouvez telecharger les images [ici](https://mega.nz/file/1lhkDJhQ#mWqVa9TpHKEHM_BTN8EfCWxjL1eFNlxYh9fGUwoRMF4).

Des fichiers CSV et de nouvelles images sont alors générées. Les chemins des fichiers CSV générés sont alors à reporter dans "data_dir" de config.json et le chemin pour les images dans "image_dir".

`python main.py -c config.json`

:warning: Deux modèles utilisent une donnée supplémentaire telle que le sexe ou l'âge: **mymodel** et **myAlexnet**. Il faut alors mettre "extended" à true pour utiliser les bons Data et Trainer.

Les autres options sont les mêmes que pour les autres procédures classiques de CNN, tel que le batch etc. Une fois toute ces étapes terminées on peut alors exécuter le programme.

`tensorboard --logdir saved/log/`

`python main.py -c config.json`

`python plot.py -c config_plot.json`

Dans config.js il faut mettre tensorboard à true , puis pour suivre l'évolution, on peut doit alors taper dans un autre terminal en parallèle:

`tensorboard --logdir saved/log/`

On peut faire la même chose mais en affichant les CSV grâce à la classe plot.py, il suffit alors de reporter dans le fichier config_plot.json, les valeurs pour pour data_loader et model, ensuite il faut reporter le bon chemin pour model_path et affiche ( qui dépend de config.js), puis on peut alors affichier en exécutant, dans le dossier visualisation:

`python plot.py -c config_plot.json`

On peut alors afficher trois figures, le loss en fonction de l'époque, l'accuracy en fonctrion de l'epoque ou encore la matrice de confusion. (Attention, la metrics est ici généré une fois le modèle chargé, bien faire correspondre config_plot.json et le config.json enregistré ).