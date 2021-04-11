import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from vgg import *


class SelectiveResNet50(nn.Module):

    def __init__(self, n_labels):
        super(SelectiveResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        num_ftrs = resnet50.fc.in_features
        self.resnet_layers = nn.Sequential(*list(resnet50.children())[:-1])
        
        # classification head (f)
        self.classification_head = nn.Linear(num_ftrs, n_labels)
        
        # selection head (g)
        self.selection_head = nn.Sequential(nn.Linear(num_ftrs, 512), 
                                            nn.ReLU(),
                                            nn.BatchNorm1d(512), #I didn't add the "/10" division because makes no sense
                                            nn.Linear(512, 1))

        # auxiliary head (h)
        self.auxiliary_head = nn.Linear(num_ftrs, n_labels)

    def forward(self, x):
        
        x = self.resnet_layers(x);
        x = torch.flatten(x, start_dim=1);
        
        classification = self.classification_head(x);
        selection = self.selection_head(x);
        auxiliary = self.auxiliary_head(x);
        
        output = torch.cat((classification, auxiliary, selection), 1);
      
        return output
        
        
        
class SelectiveVgg16_bn(nn.Module):

    def __init__(self, n_labels):
        super(SelectiveVgg16_bn, self).__init__()
        model = vgg16_bn(dropout=True, num_classes=n_labels, input_size=32)
        num_ftrs = model.classifier[4].in_features
        self.feature_extraction = nn.Sequential(*list(model.children())[:-1]);
        self.fc_first_part = nn.Sequential(*list(model.children())[-1][:-1]);
        
        
        # classification head (f)
        self.classification_head = nn.Linear(num_ftrs, n_labels)
        
        # selection head (g)
        self.selection_head = nn.Sequential(nn.Linear(num_ftrs, 512), 
                                            nn.ReLU(),
                                            nn.BatchNorm1d(512), #I didn't add the "/10" division because makes no sense
                                            nn.Linear(512, 1))

        # auxiliary head (h)
        self.auxiliary_head = nn.Linear(num_ftrs, n_labels)

    def forward(self, x):
        
        x = self.feature_extraction(x);
        x = torch.flatten(x, start_dim=1);
        x = self.fc_first_part(x)
        
        classification = self.classification_head(x);
        selection = self.selection_head(x);
        auxiliary = self.auxiliary_head(x);
        
        output = torch.cat((classification, auxiliary, selection), 1);
      
        return output