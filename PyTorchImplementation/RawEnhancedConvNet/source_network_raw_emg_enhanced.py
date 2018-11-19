import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, number_of_class, ):
        super(Net, self).__init__()
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5))
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(.5)
        
        self._conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5))
        self._pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self._batch_norm2 = nn.BatchNorm2d(64)
        self._prelu2 = nn.PReLU(64)
        self._dropout2 = nn.Dropout2d(.5)
        
        self._fc1 = nn.Linear(1024, 500)
        self._batch_norm3 = nn.BatchNorm1d(500)
        self._prelu3 = nn.PReLU(500)
        self._dropout3 = nn.Dropout(.5)
        
        
        self._output = nn.Linear(500, number_of_class)
        
        self.initialize_weights()
        
        print(self)
        
        print("Number Parameters: ", self.get_n_params())
    
    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params
    
    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
    
    def forward(self, x):
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        pool1 = self._pool1(conv1)
        conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1))))
        pool2 = self._pool2(conv2)
        flatten_tensor = pool2.view(-1, 1024)
        fc1 = self._dropout3(self._prelu3(self._batch_norm3(self._fc1(flatten_tensor))))
        output = self._output(fc1)
        return output
