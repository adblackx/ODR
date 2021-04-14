from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

class AlexNet():
	def __init__(self, num_classes, feature_extract, use_pretrained=True):
		
		self.num_classes = num_classes
		self.feature_extract = feature_extract
		self.use_pretrained = use_pretrained


	def set_parameter_requires_grad(self,model, feature_extracting):
		if feature_extracting:
			for param in model.parameters():
				param.requires_grad = False		

	def getModel(self):
		model_ft = models.alexnet(pretrained=self.use_pretrained)
		self.set_parameter_requires_grad(model_ft, self.feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,self.num_classes)
		input_size = 224

		return model_ft, input_size
