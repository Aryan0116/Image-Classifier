import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
from collections import OrderedDict

def custom_model(backbone='vgg16', hidden_units=4096, dropout=0.05):

  if backbone == 'vgg16':
    custom_model = models.vgg16(pretrained=True)
  elif backbone == 'densenet121':
    custom_model = models.densenet121(pretrained=True)
  else:
    raise ValueError("Unsupported backbone model")

  for param in custom_model.parameters():
    param.requires_grad = False
  
  custom_model.classifier = nn.Sequential(OrderedDict([
                          ('F1', nn.Linear(25088, hidden_units)),
                          ('Re', nn.ReLU()),
                          ('Drop1', nn.Dropout(p=dropout)),
                          ('F2', nn.Linear(hidden_units, 102)),
                          ('out', nn.LogSoftmax(dim=1))
                        ]))
  return custom_model