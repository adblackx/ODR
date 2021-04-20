# ODR
Ocular Disease Recognition

Pour pouvoir exécter le programme qui se trouve dans le main, il faut d'abord appliquer un preprocessing.
Le preprocessing se trouve dans le dossier utils ( ou bien à la racine en raison de bugs pour certains sur windows). Il faut d'abord décompresser l'archive du projet, et le mettre dans un dossier par exemple "data", et dans le fichier plot.py dans le main, il faut alors préciser le maths, du csv, des données bruts

Toutefois, si le preprocessing ne marche pas, voici les images:
https://mega.nz/file/1lhkDJhQ#mWqVa9TpHKEHM_BTN8EfCWxjL1eFNlxYh9fGUwoRMF4

Des fichiers csv et de nouvelles images sont alors générés. Les chemins des fichiers csv générés sont alors à reporter dans "data_dir" de config.json et le chemin pour les images dans "image_dir".

Enfin, il reste plus qu'à choisir un model, en utilisant la classe Model_Mult, on peut alors lui donner un "model_name" se trouvant dans la classe, pour obtenir un modèle.

Attention, les modèles utilisant une donnée supplémentaire tel que le sexe ou l'âge sont au nombre de deux, ce sont "mymodel" et "myAlexnet". Il faut alors mettre "extended" à true pour utiliser les bons Data et Trainer.

Les autres options sont les mêmes que pour les autres procédures classiques de CNN, tel que le batch etc.
Une fois toute ces étapes terminées on peut alorsexécuter le programme.

Le programme principale s'éxecute avec la commande : 

python main.py -c config.json


Petite nouveauté, pour que ça marche il faut installer tensorboard.
Ensuite, il faut exécuter python main.py -c config.json, dans config.js il faut mettre tensorboard à true , puis pour suivre l'évolution, on peut doit alors taper dans un autre terminal en parallèle:

tensorboard --logdir saved/log/

On peut faire la même chose mais en affichant les csv grâce à la classe plot.py, il suffit alors de reporter dans le fichier config_plot.json, les valeurs pour pour data_loader et model, ensuite il faut reporter le bon chemin pour model_path et affiche ( qui dépend de config.js), puis on peut alors affichier en exécutant, dans le dossier visualisation:

python plot.py -c config_plot.json

On peut alors afficher trois figures, le loss en focntion de l'époque, l'accuracy en fonctrion de l'epoque ou encore la matrice de confusion. (Attention, la metrics est ici généré une fois le modèle chargé, bien faire correspondre config_plot.json et le config.json enregistré ).

Il est à noter qu'on enregistre le meilleur modèle en fonction de la valid accura, qu'on enregistre le config.js, ainsi qu'un csv.

Nous mettons à disposition nos résultats contenus dans les csv:
https://mega.nz/file/UwpmBRyQ#_Ygfeoiw6DksUEi2zlJ8pm1YKQ3MywXuubloDhVyBk0

Cette architecture est tirée de https://github.com/moemen95/Pytorch-Project-Template .
De nombreuses fonctions sont tirées de pytorch et de scikit learn.
