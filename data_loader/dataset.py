  
import torch
import glob
import os
import pandas as pd
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, image_dir, transform, extended , dataAug):
        'Initialization'

        """
        to_drop = ['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus',
           'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords', 'N', 'D', 'G',
           'C', 'A', 'H', 'M', 'O', 'filepath', 'target']
        data = data.drop(columns = to_drop)
        """
        self.image_dir = image_dir
        self.extended = extended
        data = pd.read_csv(data_dir)

        self.labels = data['Label'].to_numpy()
        self.list_IDs = data['Image'].to_numpy()
        self.age = None
        if(extended):
            self.age = data['Patient Age'].to_numpy()
        print(self.labels)
        '''for i in range(len(self.labels)):
            self.labels[i] = int(self.labels[i][1])'''
        print(self.labels)
        self.transform = transform
        self.dataAug = dataAug

        #print(self.list_IDs[1:10])
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        """
        'Generates one sample of data'
        print("ICI")
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.load('data/preprocessed_images/' + ID)
        y = self.labels[ID]
        return X, y
        """

        dataAugmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
        ])

        label = self.labels[index]
        ID = self.list_IDs[index]
        #print(ID)

        img_path = self.image_dir + ID
        img = Image.open(img_path)
        if(self.dataAug):
            img = dataAugmentation(img)
        img_transformed = self.transform(img)
        

        #labels_unique = np.unique(self.labels)
        #label = self.labels[index]
        #label = np.where(labels_unique == label)[0][0]



        if not self.extended:
            return img_transformed, label
        else:
            age = int(self.age[index])
            #sex = int(self.sex[index] == "Male" )
            return img_transformed, label, age


    def getItem(self, index):
        return self.__getitem__(index)