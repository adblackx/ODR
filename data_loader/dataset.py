import torch
import glob
import os
import pandas as pd
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir,transform):
        'Initialization'

        """

        to_drop = ['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus',
           'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords', 'N', 'D', 'G',
           'C', 'A', 'H', 'M', 'O', 'filepath', 'target']

        data = data.drop(columns = to_drop)

        """

        data = pd.read_csv('data/full_df.csv')

        my_dir = data_dir+'preprocessed_images/'
        print(my_dir)
        my_list = glob.glob(os.path.join(my_dir,'*.jpg'))

        filename_list = data['filename'].to_numpy()
        labels_list = data['labels'].to_numpy()

        self.labels = labels_list
        self.list_IDs = filename_list
        self.transform = transform
        print(self.list_IDs[1:10])

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
        ID = self.list_IDs[index]
        img_path = 'data/preprocessed_images/' + ID
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        labels_unique = np.unique(self.labels)
        label = self.labels[index]
        label = np.where(labels_unique == label)[0][0]
        #print(self.label_list[idx], label)

        return img_transformed, label