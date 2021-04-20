# -*- coding: utf-8 -*-
import torch
import glob
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 
from PIL import Image,ImageOps
from torch.utils.data.dataloader import default_collate
import os,sys,inspect

from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import os
import cv2
import re

from keyphrases import diagnostic_keyphrases, diagnostic_normale_or_not


def crop(image): 
	"""
		Remove vertical black borders (the image must be already normalized)
		image: the input image
	"""

	image=np.asarray (image)
	
	sums = image.sum(axis=0)
	sums = sums.sum(axis=1)
	filter_arr = []
	for s in sums:
			if s == 0:
				filter_arr.append(False)
			else:
				filter_arr.append(True)
	image = image[:, filter_arr]
	
	# Crop to a square shape
	h = image.shape[0]
	w = image.shape[1]    
	
	if h < w:
			x = (w - h) // 2
			image = image[:, x:x+h, :]        
	elif h > w:
			x = (h - w) // 2
			image = image[x:x+w, :, :]           
	else:
			pass
	
	image = Image.fromarray (image)
	return image

def resize(image, IMG_SIZE = 512):

	"""
	resize the image to have the same size between all the images (because of different image resolutions)
	and conversion of the color of the image because by default cv2 reads image in blue color
	in the resize, the ration = 1 by default because pixel height = pixel width
	
	image: the input image
	IMG_SIZE: size of the output image

	"""
	image = np.asarray(image)
	norm_img = np.zeros(image.shape)
	# normalisation 0 ou 1
	norm_img = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)


	image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))

	image = Image.fromarray(image)
	return image

def scaleRadius(img, scale):
	"""
		rescale the images to have the same radius
		image: the input image
		scale: scale of the radius	
	"""

	radius = int(img.shape[0] / 2)
	x = img[radius, :, :].sum(1)
	r = (x > x.mean() / 10).sum() / 2
	s = scale * 1.0 / r
	return cv2.resize(img, (0, 0), fx=s , fy=s )

def graham(image, scale = 320):
	"""
		implementation of the pseudo-code from Ben Graham's study
		image: the input image
		scale: scale of the radius
	"""
	image = np.asarray (image)
	
	#  remove the borders
	# #image = preprocess_image_crop(norm_img)

	#resize the image to a given radius
	image = scaleRadius(image, scale)
	
	#we subtract the average color to map it on 50% of gray in order to better highlight the contrasts
	image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image,(0,0), scale / 30), -4, 128)
 
 
	#we remove 10% of the borders
	b = np.zeros(image.shape)

	cv2.circle(b, (int(image.shape[1] / 2), int(image.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(image)
	
	return image

def to_grayscale(image):
	"""
		switch to black and white
		image: the input image
	"""
	image = ImageOps.grayscale(image)
	return image

def generateCsvFromRaw(DATA_PATH):
	"""
		The objective of this function is to take in entry the path of a csv,
		 and to edit that if, to keep only the images which interest us
	"""

	data = pd.read_excel(DATA_PATH)
	LeftText=[]
	for i, row in data.iterrows():
			text=row['Left-Diagnostic Keywords']
			Listecle=[]
			for cle, valeur in diagnostic_keyphrases.items():
					valeur = diagnostic_keyphrases.get (cle)
					for keyword in valeur:
							if keyword in text:
									Listecle.append (cle)
						 
			Listecle=list(set(Listecle))
			LeftText.append (Listecle)
	
	data['Left Text']= LeftText
	
	
	RightText=[]
	for i, row in data.iterrows():
			text=row['Right-Diagnostic Keywords']
			Listecle=[]
			for cle, valeur in diagnostic_keyphrases.items():
					valeur = diagnostic_keyphrases.get(cle)
					for keyword in valeur:
							if keyword in text:
									Listecle.append(cle)
						 
			Listecle=list(set(Listecle))
			RightText.append (Listecle)
	
	data['Right Text']= RightText
	rf = data['Right-Fundus']
	data['filename']= data['Left-Fundus'] 
	key_columns = ['ID', 'Patient Age','Patient Sex','filename','label']
	data=data [key_columns]
	for i in range(len(RightText)):
		data = data.append({'ID': data['ID'][i], 'Patient Age': data['Patient Age'][i], 'Patient Sex': data['Patient Sex'][i], 'filename': rf[i], 'label': RightText[i]}, ignore_index=True)

	for i in range(len(data['ID'])):
		print(data['ID'][i])
	data.to_csv('full_prepro.csv', mode='a', header=key_columns, index=False, encoding='utf-8')

def getDiagnostic(text, diagnostic_keyphrases):
	"""
		text: a string containing one or more diagnoses for a given eye
		diagnostic_keyphrases: a list of string of all possible diagnoses
		
	"""
	diagnostics = re.split('， *|, *', text)
	
	l = []
	dic = {}
	for d in diagnostics:
		for key, values in diagnostic_keyphrases.items():
			if d in values and not key in l:
				l.append(key)
	return l

def generateCSV(data_path, csv_name, discard_no_labels = True, discard_more_one_label = True, filters = [], diagnostic_keyphrases = diagnostic_keyphrases):
	"""
	data_path: path of the data.xlsx file
	csv_name: the name of the csv that will be generated
	discard_no_labels : if true, removes images without labels
	discard_more_one_label: remove images that have more than 1 label
	filters: the different filters that will be applied
	diagnostic_keyphrases: a list of string of all possible diagnoses
	"""

	filetarget = "../data/ODIR-5K/csv/"
	if not os.path.exists(filetarget):
		os.makedirs(filetarget)
	data = pd.read_excel(data_path)
	
	new_data = []
	for i, row in data.iterrows():
		new_row = [row['Patient Age'], row['Patient Sex'], row['Left-Fundus']]
		labels = getDiagnostic(row['Left-Diagnostic Keywords'], diagnostic_keyphrases)
		if (not discard_no_labels or len(labels) > 0) and (not discard_more_one_label or len(labels) <= 1):
			new_row.append(labels[0])
			if len(filters) == 0:
				new_data.append(new_row)
			else:
				for f in filters:
					if f in labels:
						new_data.append(new_row)
						break
			
		new_row = [row['Patient Age'], row['Patient Sex'], row['Right-Fundus']]
		labels = getDiagnostic(row['Right-Diagnostic Keywords'], diagnostic_keyphrases)
		if (not discard_no_labels or len(labels) > 0) and (not discard_more_one_label or len(labels) <= 1):
			new_row.append(labels[0])
			if len(filters) == 0:
				new_data.append(new_row)
			else:
				for f in filters:
					if f in labels:
						new_data.append(new_row)
						break
	
	key_columns = ['Patient Age','Patient Sex', 'Image', 'Label'] 

	df = pd.DataFrame(new_data)
	print(filetarget+csv_name)
	df.to_csv(filetarget+csv_name, header=key_columns, index=False, encoding='utf-8')

def createPreprocessingFile(filepath, filetarget, prepro):
	'''
	filepath : path to the folder containing the images to be processed
	filetarger : path to the folder where to write the images (if none exists it will be created)
	prepro : function to be applied to the images
	'''
	
	print("Start Prepro Writing")
	filename = [f for f in listdir(filepath) if isfile(join(filepath, f))]
	nb_image = len(filename)
	compteur = 0
	#check if the target file exists if not create it
	if not os.path.exists(filetarget):
		os.makedirs(filetarget)
	nbFiles = len(filename)
	cpt = 0
	for img in filename:
		path = os.path.join(filepath,img)
		image = Image.open(path)
		for f in prepro: # we apply, crop, resize and graham here
			image = f(image)
		#image = image.resize((512, 512))
		path = os.path.join(filetarget,img)
		image.save(os.path.abspath(path))
		cpt += 1
		print(" [" + "#"*int(cpt/nbFiles * 20) + " "*(20-int(cpt/nbFiles * 20)) + "] ", end='')
		print("%.2f"%(cpt/nbFiles * 100) + "%", end='\r')
	print("\nEnd Prepro Writing")


if __name__=="__main__":


	#Add these three lines of code to facilitate the import of the modules
	currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	parentdir = os.path.dirname(currentdir)
	sys.path.insert(0,parentdir)


	#The code below should generate a series of csv and/or preprocessed dataset
	data_path = "data/ODIR-5K/ODIR-5K/data.xlsx"
	filetarget = "data/ODIR-5K/csv/" #do not modify the file will be created automatically
	if not os.path.exists(filetarget + "full_prepro.csv"):
		generateCSV(data_path, "full_prepro.csv")
	if not os.path.exists(filetarget + "normale_vs_diabetic.csv"):
		generateCSV(data_path, "normale_vs_diabetic.csv", filters = [0,1])
	if not os.path.exists(filetarget + "normale_vs_diabetic_vs_glaucoma.csv"):
		generateCSV(data_path, "normale_vs_diabetic_vs_glaucoma.csv", filters = [0,1,2])
	if not os.path.exists(filetarget + "normale_vs_diabetic.csv"):
		generateCSV(data_path, "normale_vs_malade.csv", diagnostic_keyphrases = diagnostic_normale_or_not)

	
	filepath =  "data/ODIR-5K/ODIR-5K/Training Images" #path to load images
	filetarget = "data/ODIR-5K/ODIR-5K/preprocess_graham" #path to save the images

	prepro = [crop, resize, graham]

	if not os.path.exists(filetarget):
		
		createPreprocessingFile(filepath, filetarget, prepro)
	elif os.path.getsize(filetarget) == 0:
		
		createPreprocessingFile(filepath, filetarget, prepro)