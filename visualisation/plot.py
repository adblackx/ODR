import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import glob

import os,sys,inspect

#Add these three lines of code to facilitate the import of the modules
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)



from pathlib import Path
import data_loader.data_loaders as module_data
from utils.util import prepare_device
import model.model_mult as model_mult
import model.mymodel as mymodel

import argparse
import collections
from parse_config import ConfigParser
from torchvision import  models
from figures import Plot
import numpy as np
from sklearn.metrics import confusion_matrix


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



def afficher(config):
	# exemple  python plot.py saved/models/Garaham/\0414_170901/metrics.csv
	
	# exemple  python plot.py saved/models/Garaham/0415_130945/metrics.csv
	# exemple  python plot.py -c config_plot.json


	aff = Plot(config['affiche'])
	#displays the loss as a function of time
	aff.printLoss()
	#displays the Accuracy as a function of time
	aff.printAccuracy()

	device, device_ids = prepare_device(config['n_gpu'])

	PATH = config['model_path']
	print("Load model from: ",PATH)

	model = config.init_obj('model', model_mult)

	model = model.getModel()
	print(model)

	if len(device_ids) > 1:
		model = torch.nn.DataParallel(model, device_ids=device_ids)

	model.load_state_dict(torch.load(PATH))
	model.eval() 

	data_loader = config.init_obj('data_loader', module_data)
	valid_data_loader = data_loader.split_validation()
	ds = data_loader.dataset

	train_Label = np.empty(len(data_loader.sampler))
	if not ds.extended:
		for i in range(len(data_loader.sampler)):
			imgSet,train_Label[i] = ds.getItem(i)
	else:
		for i in range(len(data_loader.sampler)):
			imgSet,train_Label[i],data = ds.getItem(i)	


	pred_data_loader = torch.utils.data.DataLoader(batch_size=10000, dataset=train_Label, num_workers=1)
	valid_data_loader = data_loader.split_validation()
	all_preds, train_label1 = get_all_preds(network=model, dataloader=valid_data_loader) #data_loader)

	print(len(all_preds))
	print(len(train_label1))

	preds_correct = get_num_correct(all_preds, train_label1)
	print('total correct:', preds_correct,  'sur', len(train_label1) )
	print('accuracy:', preds_correct / len(train_label1))
	 


	#displays confusion matrix
	aff.plot_confusion_matrix(cm=confusion_matrix(y_true=train_label1, y_pred=all_preds.argmax(1)), target_names = np.unique(train_label1), normalize=False)
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