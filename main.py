import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer.trainer_ram1 import Trainer
from utils.util import prepare_device
from data_loader.data_loaders import odr_data_loader
import data_loader.data_loaders as module_data

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

    logger = config.get_logger('train') # Pour tensorBoard


    '''data_loader = config.init_obj('data_loader', module_data)

    valid_data_loader = data_loader.split_validation()'''
    
    #mes modifs

    #-----------------CODE A METTRE DANS data loader et base-------------------
    data_dir= config['data_dir']

    #x_train, y_train, x_test, y_test = splitTrainTest(data_dir)

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()


    #train_loader = odr_data_loader(data_dir, x_train, y_train, 100, True, 0, 2)
    #valid_data_loader = odr_data_loader(data_dir, x_test, y_test, 100, True, 0, 2)
    #-------------------------------------

    
    #fin modifs
    #train_loader = odr_data_loader(**args)


    #------------------CODE A METTRE DANS TRAINER-------------------



    # PARTIE OK DEBUT

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    torch.manual_seed(1234)
    if device =='cuda':
        torch.cuda.manual_seed_all(1234)

    
    model = models.alexnet(pretrained=True).to(device)
    #model.train()

    logger.info(model)



    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]


    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # PARTIE OK FIN

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


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
