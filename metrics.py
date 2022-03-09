import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = F.cross_entropy(preds, labels, reduction = "none")
    mask = torch.Tensor(mask)
    mask /= torch.mean(mask)
    loss *= mask
    masked_loss = torch.mean(loss)
    return masked_loss

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct = (preds == labels)
    mask = torch.Tensor(mask)
    mask /= torch.mean(mask)
    correct = correct * mask
    masked_acc = torch.mean(correct)
    return masked_acc