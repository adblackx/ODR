# ODR
Ocular Disease Recognition

s'execute avec : 

python main.py -c config.json

L'éxecution ne marche pas pour le moment.

Nouveauté, TensorBoard ,l'installer avant bien évidemment, copier/coller le lien dans le navigateur :

tensorboard --logdir saved/log/


Pour tester 

Pour les dossiers:

* Dans model on met le modèle et/ou la fonction de cout, on devra se crée notre propre model, pour une classe qui est un réseau de neurone, elle doit hériter de nn.module
* data_loader c'est ce qui va nous charger nos données
* dans trainer on met 
* utils ce sera des codes annexes 

Pour config.json j'ai mis un exemple qui marche avec dans train.py