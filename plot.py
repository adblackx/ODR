import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import os
import sys
from pathlib import Path
import data_loader.data_loaders as module_data
from utils.util import prepare_device
import model.model_mult as model_mult
import model.mymodel as mymodel

import argparse
import collections
from parse_config import ConfigParser
from torchvision import  models

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

class Plot():
	def __init__(self,filepath):
		self.filepath = filepath
		self.data = pd.read_csv(filepath)


	def printLoss(self):
		fig, ax = plt.subplots(figsize=(10,5))
		x = self.data[['epoch']]
		y1 = self.data[['loss']]

		y2 = self.data[['val_loss']]
		ax.plot(x, y1, label="train loss")
		ax.plot(x, y2, label="validation loss")
		ax.legend()
		ax.set_title("Loss en fonction de l'epoque")
		plt.show()

	def printAccuracy(self):
		fig, ax = plt.subplots(figsize=(10,5))
		x = self.data[['epoch']]
		y1 = self.data[['accuracy']]
		y2 = self.data[['val_accuracy']]

		ax.plot(x, y1, label="train accuracy")
		ax.plot(x, y2, label="validation accuracy")
		ax.legend()
		ax.set_title("Accuracy en fonction de l'epoque")
		plt.show()



@torch.no_grad() # turn off gradients during inference for memory effieciency
def get_all_preds(network, dataloader):
	"""function to return the number of correct predictions across data set"""
	all_preds = torch.tensor([])
	true_preds = torch.tensor([])
	model = network
	if dataloader.dataset.extended:
		for batch in dataloader:
			images, labels,data = batch
			#print(data.size())
			#print(data)
			preds = model(images,data) # get preds
			all_preds = torch.cat((all_preds, preds), dim=0) # join along existing axis
			true_preds = torch.cat((true_preds, labels), dim=0)
	else:
		for batch in dataloader:
			images, labels, = batch
			preds = model(images) # get preds
			all_preds = torch.cat((all_preds, preds), dim=0) # join along existing axis
			true_preds = torch.cat((true_preds, labels), dim=0)
		
	return all_preds, true_preds

def get_num_correct(preds, labels):
	return preds.argmax(dim=1).eq(labels).sum().item()


def plot_confusion_matrix(cm,
						  target_names,
						  title='Confusion matrix',
						  cmap=None,
						  normalize=True):
	"""
	given a sklearn confusion matrix (cm), make a nice plot

	Arguments
	---------
	cm:           confusion matrix from sklearn.metrics.confusion_matrix

	target_names: given classification classes such as [0, 1, 2]
				  the class names, for example: ['high', 'medium', 'low']

	title:        the text to display at the top of the matrix

	cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
				  see http://matplotlib.org/examples/color/colormaps_reference.html
				  plt.get_cmap('jet') or plt.cm.Blues

	normalize:    If False, plot the raw numbers
				  If True, plot the proportions

	Usage
	-----
	plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
															  # sklearn.metrics.confusion_matrix
						  normalize    = True,                # show proportions
						  target_names = y_labels_vals,       # list of names of the classes
						  title        = best_estimator_name) # title of graph

	Citiation
	---------
	http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

	"""
	import matplotlib.pyplot as plt
	import numpy as np
	import itertools

	accuracy = np.trace(cm) / np.sum(cm).astype('float')
	misclass = 1 - accuracy

	if cmap is None:
		cmap = plt.get_cmap('Blues')

	plt.figure(figsize=(15, 15))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=45)
		plt.yticks(tick_marks, target_names)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


	thresh = cm.max() / 1.5 if normalize else cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if normalize:
			plt.text(j, i, "{:0.4f}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")
		else:
			plt.text(j, i, "{:,}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")


	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
	plt.show()

def afficher(config):
	# exemple  python plot.py saved/models/Garaham/\0414_170901/metrics.csv
	
	#DECOMMENTER POUR AFFICHER TRAIN EN FONCTION DE LOSS
	# exemple  python plot.py saved/models/Garaham/0415_130945/metrics.csv
	# exemple  python plot.py -c config_plot.json
	
	"""
	print("Hello, world!")
	aff = Plot(config['affiche'])
	aff.printLoss()
	aff.printAccuracy()
	"""

	aff = Plot(config['affiche'])
	aff.printLoss()
	aff.printAccuracy()

	model = models.alexnet(pretrained=True)

	device, device_ids = prepare_device(config['n_gpu'])

	PATH = config['model_path']
	print(PATH)

	model = config.init_obj('model', model_mult)

	model = model.getModel()
	print(model)

	if len(device_ids) > 1:
		model = torch.nn.DataParallel(model, device_ids=device_ids)
		print("AAAAAAAAAAAAAAAAAA")

	print(torch.load(PATH))
	model.load_state_dict(torch.load(PATH))
	model.eval() # voir doc pk

	data_loader = config.init_obj('data_loader', module_data)
	valid_data_loader = data_loader.split_validation()
	ds = data_loader.dataset
	#trainSet = np.empty(len(data_loader.sampler))
	print("debut")
	# IL FAUT AVOIR UNE CORRESPONDANCE DE TAILLE ENTRE LES LABELS ET CE QUE DONNE DATA LOADER
	train_Label = np.empty(len(data_loader.sampler))
	if not ds.extended:
		for i in range(len(data_loader.sampler)):
			imgSet,train_Label[i] = ds.getItem(i)
	else:
		for i in range(len(data_loader.sampler)):
			imgSet,train_Label[i],data = ds.getItem(i)	

	print("fin")


	pred_data_loader = torch.utils.data.DataLoader(batch_size=10000, dataset=train_Label, num_workers=1)
	valid_data_loader = data_loader.split_validation()
	all_preds, train_label1 = get_all_preds(network=model, dataloader=valid_data_loader) #data_loader)

	#train_label1 = torch.from_numpy(train_Label)
	print(len(all_preds))
	print(len(train_label1))

	preds_correct = get_num_correct(all_preds, train_label1)
	print('total correct:', preds_correct,  'sur', len(train_label1) )
	print('accuracy:', preds_correct / len(train_label1))
	 



	plot_confusion_matrix(cm=confusion_matrix(y_true=train_label1, y_pred=all_preds.argmax(1)), target_names = np.unique(train_label1), normalize=False)
	plt.savefig("testconf.png")





	
	



if __name__ == '__main__': #ne pas modifier cette fonction
	args = argparse.ArgumentParser(description='PyTorch Template')
	args.add_argument('-c', '--config', default=None, type=str,
					  help='config file path (default: None)')
	args.add_argument('-r', '--resume', default=None, type=str,
					  help='path to latest checkpoint (default: None)')
	args.add_argument('-d', '--device', default=None, type=str,
					  help='indices of GPUs to enable (default: all)')

	# custom cli options to modify configuration from default values given in json file.
	CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
	options = [
		CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
		CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
	]
	config = ConfigParser.from_args(args, options)
	#print(torch.cuda.is_available()) # affiche si cuda est dispo ou non
	afficher(config)