# ODR
Ocular Disease Recognition

Le programme principale s'éxecute avec la commande : 

python main.py -c config.json


Petite nouveauté, pour que ça marche il faut installer tensorboard.
Ensuite, il faut exécuter python main.py -c config.json, dans config.js il faut mettre tensorboard à trye , puis pour suivre l'évolution, on peut doit alors taper dans un autre terminal en parallèle:

tensorboard --logdir saved/log/

On peut faire la même chose mais en affichant les csv grâce à la classe plot.py, il suffit alors de reporter dans le fichier config_plot.json, les valeurs pour pour data_loader et model, ensuite il faut reporter le bon chemin pour model_path et affiche ( qui dépend de config.js), puis on peut alors affichier en exécutant:

python plot.py -c config_plot.json

On peut alors afficher trois figures, le loss en focntion de l'époque, l'accuracy en fonctrion de l'epoque ou encore la matrice de confusion.

Cette architecture est tirée de https://github.com/moemen95/Pytorch-Project-Template .
De nombreuses fonctions sont tirées de pytorch et de scikit learn.

