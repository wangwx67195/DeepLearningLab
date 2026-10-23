import torch
import torch.nn as nn

# Create loss function
def create_loss_fn():
    return nn.CrossEntropyLoss()