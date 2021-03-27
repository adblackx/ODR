import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

    #https://pytorch.org/docs/stable/nn.functional.html
    # on peut tester les autres classes qui se trouvent dedans