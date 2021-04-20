import torch
import glob
import os
import numpy as np
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, IMG_DIR, transform):
        'Initialization'

        """
        to_drop = ['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus',
           'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords', 'N', 'D', 'G',
           'C', 'A', 'H', 'M', 'O', 'filepath', 'target']
        data = data.drop(columns = to_drop)
        """

        data = pd.read_csv('full_df.csv')
  
        my_dir = data_dir+'preprocessed_images/'
  
        my_list = glob.glob(os.path.join(my_dir,'*.jpg'))
        
        ListeMaladie=["['N']", "['D']", "['G']", "['C']","['A']", "['H']","['M']","['O']"]

        left_labels_list=[]
        right_labels_list=[]
        left_name_list=[]
        right_name_list =[]
    
        for i in range (len (data)):
            if data['Left Text'][i] in ListeMaladie:
                left_labels_list.append(data['Left Text'][i])
                left_name_list.append(data['Left-Fundus'][i])
                
            if data['Right Text'][i] in ListeMaladie:
                right_labels_list.append (data['Right Text'][i])
                right_name_list.append (data['Right-Fundus'][i])
            
        filename_list = np.concatenate((right_name_list, left_name_list), axis=None)
        labels_list = np.concatenate((right_labels_list, left_labels_list), axis=None)
        
        self.labels = labels_list
        self.list_IDs = filename_list
        self.transform = transform


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        img_path = IMG_DIR + ID
        #print("processing", img_path)
        img = Image.open(img_path)

        img_transformed = self.transform(img)


        labels_unique = np.unique(self.labels)
      
        label = self.labels[index]
        label = np.where(labels_unique == label)[0][0]
        
        X=img_transformed
        y=label
   
        return X, y

