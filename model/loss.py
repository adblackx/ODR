import torch.nn.functional as F
import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target):
	return F.cross_entropy(output, target) #nn.CrossEntropyLoss(output, target)

    #F.nll_loss(output, target)

    #https://pytorch.org/docs/stable/nn.functional.html
    # on peut tester les autres classes qui se trouvent dedans