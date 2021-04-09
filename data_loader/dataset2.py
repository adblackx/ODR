import torch
import glob
import os
import pandas as pd
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, filename_list, labels_list, transform):
        'Initialization'

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

def countNbElementByClasses(labels, msg = ""):
    labels = np.array(labels)
    print(msg)
    for i in np.unique(labels):
        print("nb classes" , i , " : ", np.count_nonzero(labels == i))

def splitTrainTest(filepath, test_size=0.05, test_equal_repartition = True, Shuffle = False):
    
    data = pd.read_csv(filepath)
        
    #On extrait les labels et le filename de notre dataset
    filename_list = data['filename'].to_numpy()
    labels_list = data['labels'].to_numpy()
    size = len(labels_list)
    
    #On transforme les labels en chiffres
    labels_unique = np.unique(labels_list).tolist()
    labels = []
    for i in labels_list:
        labels.append(labels_unique.index(i))
    labels = np.array(labels)
    
    ind = np.arange(size)
    if(Shuffle):
        np.random.shuffle(ind)
    
    
    #filename_list = [my_dir]*len(filename_list) + filename_list #Pour avoir le chemin exact de nos images et pas juste le nom
    
    if(test_equal_repartition):
        nb_img_by_classes = int(test_size * len(labels)/len(np.unique(labels)))
        
        print("image by classes :", nb_img_by_classes)
        nb_img = np.zeros(len(np.unique(labels))) + nb_img_by_classes
        
        test_list, y_test, train_list, y_train = [],[],[],[]
        
        for i in ind:
            idx = labels[i]
            if nb_img[idx] > 0:
                nb_img[idx]-=1
                test_list.append(filename_list[i])
                y_test.append(labels[i])
            else:
                train_list.append(filename_list[i])
                y_train.append(labels[i])
    else:
        #On va créer notre ensemble d'netrainement et de test
        #ind = np.arange(size)
        train_size = 1.0 - test_size
        end_train = int(train_size*len(filename_list))
        
        print(ind)
        index_train = ind[:end_train]
        index_test = ind[end_train:]
        
        train_list = filename_list[index_train]
        test_list = filename_list[index_test]
        
        #Ci-dessous nos labels
        y_train = labels[index_train]
        y_test = labels[index_test]
    
    print("train size:", len(y_train),"-- test size:", len(y_test) )
    countNbElementByClasses(y_train, msg = "[TRAIN SET]")
    countNbElementByClasses(y_test, msg = "[TEST SET]")
    
    return train_list, y_train, test_list, y_test
    
    
    
    
    
    
    
    
    
    
    