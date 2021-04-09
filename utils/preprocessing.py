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
#from base.base_data_loader import BaseDataLoader
#from data_loader.dataset import Dataset
from torch.utils.data.dataloader import default_collate
#from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import os
import cv2

def preprocess_image_Crop(image): 
    image=np.asarray (image)
    # Remove vertical black borders (the image must be already normalized)
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
        x = (w - h)//2
        image = image[:, x:x+h, :]        
    elif h > w:
        x = (h - w)//2
        image = image[x:x+w, :, :]           
    else:
        pass
    
    image=Image.fromarray (image)
    return image

def preprocess_image_Resize(image, IMG_SIZE = 512):
    image=np.asarray (image)
    norm_img = np.zeros(image.shape)
    # normalisation 0 ou 1
    norm_img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    # redimension de l'image pour avoir meme dimension entre toutes les images (à cause image resolution differentes)
    # et conversion de la couleur de l'image car par defaut cv2 lit image en couleur bleue
    # dans le resize, le ration = 1 par defaut car hauteur pixel = largeur pixel
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))

    image = Image.fromarray (image)
    return image

def scaleRadius (img, scale):
    toto=int(img.shape[0]/2)
    x=img[toto,:,:].sum (1)
    r=(x>x.mean()/10).sum()/2
    s=scale * 1.0/r
    return cv2.resize (img,(0,0), fx=s , fy=s )

# implementation pseudo-code Ben Graham
def preprocess_image_Ben (image, scale = 320):
    
    image = np.asarray (image)
    
    # enleve les bordures
    # #image = preprocess_image_crop(norm_img)

    #redimensionnement de l'image à un rayon donné
    image=scaleRadius (image, scale)
    
    #on soustrait la couleur moyenne pour la mapper sur 50% de gris de façon à mieux faire ressortir les constrastes
    image=cv2.addWeighted (image ,4,cv2.GaussianBlur (image,(0,0),scale/30),-4 ,128)
   
   
    #on enleve 10% des bordures
    b=np.zeros(image.shape)

    cv2.circle(b,(int(image.shape[1]/2), int(image.shape[0]/2)),int(scale * 0.9),(1,1,1),-1,8,0)
    #image=image*b+128*(1-b)
    
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=Image.fromarray (image)
    
    return image

def to_grayscale(image):
    #suppression des bordures
    # passage en noir et blanc
    #image=np.asarray (image)
    image=ImageOps.grayscale(image)
    #image=Image.fromarray (image)
    return image

def generateCsvFromRaw(DATA_PATH):
    data = pd.read_excel(DATA_PATH)

    diagnostic_keyphrases = {'N': ['normal fundus'], 
     'D': ['nonproliferative retinopathy',
      'non proliferative retinopathy','mild nonproliferative retinopathy',
      'proliferative retinopathy','diabetic retinopathy'],
     'G': ['glaucoma'],
     'C': ['cataract'],
     'A': ['age-related macular degeneration'],
     'H': ['hypertensive'],
     'M': ['myopi'],
     'O': ['macular epiretinal membrane',
      'epiretinal membrane',
      'drusen',
      'myelinated nerve fibers',
      'vitreous degeneration',
      'refractive media opacity',
      'spotted membranous change',
      'tessellated fundus',
      'maculopathy',
      'chorioretinal atrophy',
      'branch retinal vein occlusion',
      'retinal pigmentation',
      'white vessel',
      'post retinal laser surgery',
      'epiretinal membrane over the macula',
      'retinitis pigmentosa',
      'central retinal vein occlusion',
      'optic disc edema',
      'post laser photocoagulation',
      'retinochoroidal coloboma',
      'atrophic change',
      'optic nerve atrophy',
      'old branch retinal vein occlusion',
      'depigmentation of the retinal pigment epithelium',
      'chorioretinal atrophy with pigmentation proliferation',
      'central retinal artery occlusion',
      'old chorioretinopathy',
      'pigment epithelium proliferation',
      'retina fold',
      'abnormal pigment ',
      'idiopathic choroidal neovascularization',
      'branch retinal artery occlusion',
      'vessel tortuosity',
      'pigmentation disorder',
      'rhegmatogenous retinal detachment',
      'macular hole',
      'morning glory syndrome',
      'atrophy',
      'laser spot',
      'arteriosclerosis',
      'asteroid hyalosis',
      'congenital choroidal coloboma',
      'macular coloboma',
      'optic discitis',
      'oval yellow-white atrophy',
      'wedge-shaped change',
      'wedge white line change',
      'retinal artery macroaneurysm',
      'retinal vascular sheathing',
      'suspected abnormal color of  optic disc',
      'suspected retinal vascular sheathing',
      'suspected retinitis pigmentosa',
      'silicone oil eye']}
    
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
            valeur = diagnostic_keyphrases.get (cle)
            for keyword in valeur:
                if keyword in text:
                    Listecle.append (cle)
               
        Listecle=list(set(Listecle))
        RightText.append (Listecle)
    
    data['Right Text']= RightText
    key_columns = ['ID', 'Patient Age','Patient Sex','Left-Fundus','Right-Fundus','N','D','G','C','A','H','M','O','Left Text','Right Text'] 
    data=data [key_columns]
    data.head()
    
    df=data
    df.to_csv('full_df.csv', mode='a', header=key_columns, index=False, encoding='utf-8')