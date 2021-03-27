import torch

#output ce sont les predictions
#target ce sont les vrais classes
def accuracy(output, target):
    with torch.no_grad(): # desactivation les variable auront require_grad = false meme sis les entree sont true
        pred = torch.argmax(output, dim=1) # https://pytorch.org/docs/stable/generated/torch.argmax.html
        # ce fait fait argmax c'est que pour chaque donnés on prends sa plus forte probabilité d'apparetenir a une classe
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

#https://pytorch.org/docs/stable/generated/torch.topk.html
def top_k_acc(output, target, k=3): # meme fonction sauf que la on travaille sur un tensor
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)