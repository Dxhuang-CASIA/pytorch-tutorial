import torch.nn as nn

def model(input_size, num_classes):
    return nn.Linear(input_size, num_classes)