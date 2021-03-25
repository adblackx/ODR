import torch.nn as nn # on est quasi sûr de l'utiliser
import torch.nn.functional as F

# on peut ajouer des fichiers pour la fonction de cout et la metric 

class odr_model():

	def __init__(self, num_classes=10):
		# faire quelque chose
 