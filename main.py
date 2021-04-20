import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
import model.model_mult as model_mult
from parse_config import ConfigParser
from utils.util import prepare_device
import data_loader.data_loaders as module_data1

from torchvision import datasets, models, transforms, utils
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


# python main.py -c config.json


def main(config):
    print("DEBUT DU PROGRAMME")

    # fix random seeds for reproducibility
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    if config['mult_data']:
        from trainer.trainer_mult import Trainer
    else:
        from trainer.trainer import Trainer
    

    device, device_ids = prepare_device(config['n_gpu'])
    #Ce code marche et oui hahaha
    logger = config.get_logger('train') # Pour tensorBoard
    train_loader = config.init_obj('data_loader', module_data1)

    valid_data_loader = train_loader.split_validation()
    


    
    #model = models.alexnet(pretrained=True).to(device)
    print("DEBUT")
    model = config.init_obj('model', model_mult)
    print("DEBUT")
    model = model.getModel()
    model = model.to(device)
    model.train() # deja fait dans trainer
    print("FIIIIIN")

    #model = models.resnet18(pretrained=True)
    #model = models.alexnet(pretrained=True)
    #print(model)
    #model.fc = nn.Linear(512, 2)
    #model = model.to(device)

    logger.info(model)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)



    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]


    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # PARTIE OK FIN

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_loader,
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
