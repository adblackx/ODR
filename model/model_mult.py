from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from model.myAlexnet import myAlexnet
from model.alexnet_custom import alexnetcustom
from model.mymodel import MyModel
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

class Model_Mult():
	def __init__(self, num_classes, feature_extract, use_pretrained=True, model_name="alexnet"):
		
		self.num_classes = num_classes
		self.feature_extract = feature_extract
		self.use_pretrained = use_pretrained
		self.model_name = model_name
		self.model = self.initialize_model(model_name, num_classes, feature_extract, use_pretrained)



	def set_parameter_requires_grad(self,model, feature_extracting):
		if feature_extracting:
			for param in model.parameters():
				param.requires_grad = False


	def initialize_model(self,model_name, num_classes, feature_extract, use_pretrained=True):
		# Initialize these variables which will be set in this if statement. Each of these
		#   variables is model specific.
		model_ft = None
		input_size = 0

		if model_name == "resnet":
			""" Resnet18
			"""
			model_ft = models.resnet18(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, feature_extract)
			num_ftrs = model_ft.fc.in_features
			model_ft.fc = nn.Linear(num_ftrs, num_classes)
			input_size = 224

		elif model_name == "alexnet_custom":
			""" Resnet18
			"""
			model_ft = alexnetcustom(pretrained=use_pretrained)
			print(model_ft)
			self.set_parameter_requires_grad(model_ft, feature_extract)
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
			input_size = 224

		elif model_name == "mymodel":
			""" Resnet18
			"""
			size_out = num_classes*100
			model_cnn = self.initialize_model("alexnet", size_out, feature_extract, use_pretrained)
			model_ft = MyModel(model_cnn,size_out,num_classes)



		elif model_name == "alexnet_custom":
			""" Resnet18
			"""
			model_ft = alexnetcustom(pretrained=use_pretrained)
			print(model_ft)
			self.set_parameter_requires_grad(model_ft, feature_extract)
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
			input_size = 224

			
		elif model_name == "alexnet":
			""" Alexnet
			"""
			model_ft = models.alexnet(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, feature_extract)
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
			input_size = 224


		elif model_name == "vgg":
			""" VGG11_bn
			"""
			model_ft = models.vgg11_bn(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, feature_extract)
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
			input_size = 224

		elif model_name == "squeezenet":
			""" Squeezenet
			"""
			model_ft = models.squeezenet1_0(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, feature_extract)
			model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
			model_ft.num_classes = num_classes
			input_size = 224

		elif model_name == "densenet":
			""" Densenet
			"""
			model_ft = models.densenet161(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, feature_extract)
			num_ftrs = model_ft.classifier.in_features
			model_ft.classifier = nn.Linear(num_ftrs, num_classes)
			input_size = 224

		elif model_name == "inception":
			""" Inception v3
			Be careful, expects (299,299) sized images and has auxiliary output
			"""
			model_ft = models.inception_v3(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, feature_extract)
			# Handle the auxilary net
			num_ftrs = model_ft.AuxLogits.fc.in_features
			model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
			# Handle the primary net
			num_ftrs = model_ft.fc.in_features
			model_ft.fc = nn.Linear(num_ftrs,num_classes)
			input_size = 299

		elif model_name == "myalexnet":
			model_ft = myAlexnet(pretrained=use_pretrained)
			print(model_ft)
			self.set_parameter_requires_grad(model_ft, feature_extract)
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
			input_size = 224

		else:
			print("Invalid model name, exiting...")
			exit()

		return model_ft					

	def getModel(self):
		return self.model 
