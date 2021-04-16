import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

import torch.nn.functional as F

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block

class MyModel(nn.Module):
    def __init__(self,model_cnn,size_out,num_classes):
        
        super(MyModel, self).__init__()
        self.cnn = model_cnn # un model pre entrainé au paravant
        
        self.fc1 = nn.Linear((size_out+1)  , 60) # 512 = btachsize et 20 c'est moi qui ai choisit au pif mdr
        self.fc2 = nn.Linear(60, num_classes)
        
    def forward(self, image, tab):
        x1 = self.cnn(image)
        x2 = tab

        x2 =  torch.unsqueeze(x2, 1)
        x = torch.cat((x2, x1), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x

