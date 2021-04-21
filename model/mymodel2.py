import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

import torch.nn.functional as F

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block

class MyModel2(nn.Module):
    def __init__(self,model_cnn,size_out,num_classes):
        
        super(MyModel2, self).__init__()
        self.cnn = model_cnn # un model pre entrainé au paravant
        self.classifier = nn.Linear(size_out +1, num_classes)

        #self.fc1 = nn.Linear((num_classes+1)  , 60) # 512 = btachsize et 20 c'est moi qui ai choisit au pif mdr
        #self.fc2 = nn.Linear(60, num_classes)
        
    def forward(self, image, tab):
        """x1 = self.cnn(image)
        x2 = tab

        x2 =  torch.unsqueeze(x2, 1)
        print(x1.size())
        x = torch.cat((x2, x1), dim=1)
        print(x.size())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)"""

        x1 = self.cnn(image)
        x2 = tab
        print(x1.size())
        print(x2.size())
        
        x2 =  torch.unsqueeze(x2, 1)
        features = torch.cat([x1, x2], dim=1)




        return self.classifier(features)

