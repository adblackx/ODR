# ODR
Ocular Disease Recognition

s'execute avec : 

python main.py -c config.json

L'éxecution ne marche pas pour le moment.

Petite nouveauté, pour que ça marche il faut installer tensorboard avec pip install tensorflow .
IEnsuite, il faut exécuter python main.py -c config.json ,  puis pour suivre l'évolution, on peut doit alors taper dans un autre terminal en parallèle:

tensorboard --logdir saved/log/

Copier coller le lien dans un navigateur, et suivre l'évolution en direct


Pour tester 

Pour les dossiers:

* Dans model on met le modèle et/ou la fonction de cout, on devra se crée notre propre model, pour une classe qui est un réseau de neurone, elle doit hériter de nn.module
* data_loader c'est ce qui va nous charger nos données
* dans trainer on met 
* utils ce sera des codes annexes 

Pour config.json j'ai mis un exemple qui marche avec dans train.py