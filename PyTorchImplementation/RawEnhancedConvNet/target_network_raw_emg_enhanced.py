import numpy as np

import torch
import torch.nn as nn

from ScaleLayer import ScaleLayer


class SourceNetwork(nn.Module):
    def __init__(self, number_of_class, dropout_rate=.5):
        super(SourceNetwork, self).__init__()
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5))
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(dropout_rate)
    
        self._conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5))
        self._pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self._batch_norm2 = nn.BatchNorm2d(64)
        self._prelu2 = nn.PReLU(64)
        self._dropout2 = nn.Dropout2d(dropout_rate)
    
        self._fc1 = nn.Linear(1024, 500)
        self._batch_norm3 = nn.BatchNorm1d(500)
        self._prelu3 = nn.PReLU(500)
        self._dropout3 = nn.Dropout(dropout_rate)
    
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
    
class TargetNetwork(nn.Module):
    def __init__(self, number_of_class, weights_pre_trained_convnet, dropout=.5):
        super(TargetNetwork, self).__init__()
        self._target_conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5))
        self._target_pool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self._target_batch_norm1 = nn.BatchNorm2d(32)
        self._target_prelu1 = nn.PReLU(32)
        self._target_dropout1 = nn.Dropout2d(dropout)

        self._target_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5))
        self._target_pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self._target_batch_norm2 = nn.BatchNorm2d(64)
        self._target_prelu2 = nn.PReLU(64)
        self._target_dropout2 = nn.Dropout2d(dropout)

        self._target_fc1 = nn.Linear(1024, 500)
        self._target_batch_norm3 = nn.BatchNorm1d(500)
        self._target_prelu3 = nn.PReLU(500)
        self._target_dropout3 = nn.Dropout(dropout)

        self._target_output = nn.Linear(500, number_of_class)
        
        self._source_weight_merge_1 = ScaleLayer((1, 32, 1, 1))
        self._source_weight_merge_2 = ScaleLayer((1, 64, 1, 1))
        self._source_weight_merge_3 = ScaleLayer((1, 500))
        
        self.initialize_weights()
        
        # Start with the pre-trained model
        #Change to seven for the new gesture target network (number of class)
        pre_trained_model = SourceNetwork(number_of_class=7, dropout_rate=dropout)
        self._added_source_network_to_graph = nn.Sequential(*list(pre_trained_model.children()))
        
        print("Number Parameters: ", self.get_n_params())

        # Load the pre-trained model weights (Source Network)
        pre_trained_model.load_state_dict(weights_pre_trained_convnet)

        # Freeze the weights of the pre-trained model so they do not change during training of the target network
        # (except for the BN layers that will be trained as normal).
        for child in pre_trained_model.children():
            if isinstance(child, nn.BatchNorm2d) is False:
                for param in child.parameters():
                    param.requires_grad = False

        self._source_network = pre_trained_model._modules
        print(self._source_network.keys())
        
    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        conv1 = self._source_network["_dropout1"](self._source_network["_prelu1"](
            self._source_network["_batch_norm1"](self._source_network["_conv1"](x))))
        conv1_target = self._target_dropout1(self._target_prelu1(self._target_batch_norm1(self._target_conv1(x))))
        conv1_target_added = conv1_target + self._source_weight_merge_1(conv1)
        
        pool1 = self._source_network["_pool1"](conv1)
        pool1_target = self._target_pool1(conv1_target_added)
        
        conv2 = self._source_network["_dropout2"](self._source_network["_prelu2"](
            self._source_network["_batch_norm2"](self._source_network["_conv2"](pool1))))
        conv2_target = self._target_dropout2(self._target_prelu2(self._target_batch_norm2(self._target_conv2(pool1_target))))
        conv2_target_added = conv2_target + self._source_weight_merge_2(conv2)
        
        pool2 = self._source_network["_pool2"](conv2)
        pool2_target = self._target_pool2(conv2_target_added)
        
        # print(np.shape(pool2))
        flatten_tensor = pool2.view(-1, 1024)
        flatten_tensor_target = pool2_target.view(-1, 1024)
    
        fc1 = self._source_network["_dropout3"](self._source_network["_prelu3"](
            self._source_network["_batch_norm3"](self._source_network["_fc1"](flatten_tensor))))
        fc1_target = self._target_dropout3(
            self._target_prelu3(self._target_batch_norm3(self._target_fc1(flatten_tensor_target))))
        fc1_target_added = fc1_target + self._source_weight_merge_3(fc1)
        output = self._target_output(fc1_target_added)
        return output