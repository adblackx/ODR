  
import torch
import glob
import os
import pandas as pd
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    """
    We have created our own DataSet class to be able to iterate on our data according to the label, 
    an image but also according to a third data, which can be the age or the sex
    """
    def __init__(self, data_dir, image_dir, transform, extended , dataAug):
        'Initialization'


        self.image_dir = image_dir
        self.extended = extended
        data = pd.read_csv(data_dir)

        self.labels = data['Label'].to_numpy()
        self.list_IDs = data['Image'].to_numpy()
        self.age = None
        self.sex = None
        if(extended):
            self.age = data['Patient Age'].to_numpy()
            self.sex = data['Patient Sex'].to_numpy()
        print(self.labels)

        print(self.labels)
        self.transform = transform
        self.dataAug = dataAug

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        """
            Index: index of the item
            self.dataAug: if true apply dataAugmentation transformations to the image
            self.extended: if true returns img_transformed, label AND age (or sex)

        """

        dataAugmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.RandomAffine(10, shear=50),
            transforms.Pad(25, fill=0, padding_mode="reflect")
        ])

        label = self.labels[index]
        ID = self.list_IDs[index]

        img_path = self.image_dir + ID
        img = Image.open(img_path)

        if(self.dataAug):
            img = dataAugmentation(img)
        img_transformed = self.transform(img)

        if not self.extended:
            return img_transformed, label
        else:
            age = (self.age[index]).astype(np.float32)
            #sex = int(self.sex[index] == "Male" )
            return img_transformed, label, age


    def getItem(self, index):
        return self.__getitem__(index)