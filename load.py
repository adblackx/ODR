import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders_prepross2 as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer.trainer import Trainer
from utils.util import prepare_device
from data_loader.data_loaders_prepross2 import odr_data_loader
from data_loader.dataset2 import splitTrainTest

from torchvision import datasets, models, transforms, utils
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


# python load.py -c config.json

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    print("DEBUT DU PROGRAMME")

    '''data_loader = config.init_obj('data_loader', module_data)

    valid_data_loader = data_loader.split_validation()'''
    
    #mes modifs
    data_dir='data/full_df.csv'

    x_train, y_train, x_test, y_test = splitTrainTest(filepath = data_dir)
    
    data_dir = "data/preprocessed_images/"
    train_loader = module_data.odr_data_loader(data_dir, x_train, y_train, 100, True, 0, 2)
    valid_data_loader = module_data.odr_data_loader(data_dir, x_test, y_test, 100, True, 0, 2)
    
    #fin modifs
    #train_loader = odr_data_loader(**args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    torch.manual_seed(1234)
    if device =='cuda':
        torch.cuda.manual_seed_all(1234)

    
    model = models.alexnet(pretrained=True).to(device)
    model.train()

    lr = 0.01 # learning_rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    epochs = 10


    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:

            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = ((output.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc/len(data_loader)
            epoch_loss += loss/len(data_loader)

        
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        

        with torch.no_grad():
                epoch_val_accuracy=0
                epoch_val_loss =0
                for data, label in valid_data_loader:
                    data = data.to(device)
                    label = label.to(device)
                    
                    val_output = model(data)
                    val_loss = criterion(val_output,label)
                    
                    
                    acc = ((val_output.argmax(dim=1) == label).float().mean())
                    epoch_val_accuracy += acc/ len(valid_data_loader)
                    epoch_val_loss += val_loss/ len(valid_data_loader)
                    
                print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))
        


    print("FIN DU PROGRAMME")


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
    main(config)
