import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Your TensorFlow code here

import torch.nn as nn
from transformers import Swinv2Model

class DefineSwinMultilabel(nn.Module):
    def __init__(self, swin_model=None, num_classes=2, activation='sigmoid'):
        super(DefineSwinMultilabel, self).__init__()
        self.swin_model = None
        if swin_model is None:
            print("Swin model default created!")    
            self.swin_model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        else: 
            self.swin_model = swin_model

        self.num_classes = num_classes
        self.linear = nn.Linear(in_features=768, out_features=self.num_classes)
        self.activate = None
        if activation == 'sigmoid':  # multi-label classification
            self.activate = nn.Sigmoid()
        elif activation == 'softmax':
            self.activate = nn.Softmax(dim=1) 
        else:
            raise ValueError('Activation function not supported')
        
    def forward(self, pixel_values):
        x = self.swin_model(pixel_values).pooler_output # [1, 768]
        x = self.linear(x) # [1, 14]
        x = self.activate(x)
        return x.squeeze(0)