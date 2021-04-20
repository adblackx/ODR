import torch

#output ce sont les predictions
#target ce sont les vrais classes
def accuracy(output, target):
    """
    Caculate the accuracy of a modele
    output:  predictions of the model
    target:  true label 
    """
    with torch.no_grad(): 
        pred = torch.argmax(output, dim=1) 
        #what makes argmax is that for each data we take its highest probability to belong to a class
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3): 
    """
    Caculate the accuracy of a modele
    output:  predictions of the model
    target:  true label 
    same function except that we work on a tensor
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)