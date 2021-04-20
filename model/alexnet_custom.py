import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Any
import torch.nn.functional as F
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNetCustom(nn.Module):

	"""	
	An alexNet that takes as input an image, a label and an additional data like age or sex
	This is a new feature added by us.

	We added two linear functions, what we do here is that we 
	concatenate the product of AlexNet (here not trained, we don't use the transfert learning in fact here)

	"""

	def __init__(self, num_classes: int = 1000) -> None:
		super(AlexNetCustom, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

		self.fc1 = nn.Linear(3, 512)
		self.fc2 = nn.Linear(512, num_classes)



	def forward(self, x1: torch.Tensor, x2) -> torch.Tensor:
		print(x1.size())
		x1 = self.features(x1)
		x1 = self.avgpool(x1)
		x1 = torch.flatten(x1, 1)
		print(x1.size())
		x1 = self.classifier(x1)
		x2 =  torch.unsqueeze(x2, 1)
		x = torch.cat((x1, x2), dim=1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x) 

		return x


def alexnetcustom(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNetCustom:
    """AlexNet model architecture from the  documentation of pytorch

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetCustom(**kwargs)
    if not pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model