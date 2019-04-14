import torch
import torch.nn as nn
import torch.nn.functional as F
from Pytorch_implementation.CWT.McDropout import McDropout
import numpy as np

class Net(nn.Module):
    def __init__(self, number_of_class, batch_size=300, number_of_channel=8, learning_rate=0.02, dropout=.5):
        super(Net, self).__init__()

        self._input_batch_norm = nn.BatchNorm2d(12, eps=1e-4)
        #self._input_prelu = pelu((1, 12, 1, 1))
        self._input_prelu = nn.PReLU(12)

        self._list_conv1_first_part = []
        self._list_conv2_first_part = []
        self._first_part_dropout1 = []
        self._first_part_dropout2 = []
        self._first_part_relu1 = []
        self._first_part_relu2 = []
        self._first_part_batchnorm1 = []
        self._first_part_batchnorm2 = []
        for i in range(4):
            self._list_conv1_first_part.append(nn.Conv2d(3, 8, kernel_size=3))
            self._list_conv2_first_part.append(nn.Conv2d(8, 12, kernel_size=3))

            self._first_part_dropout1.append(McDropout())
            self._first_part_dropout2.append(McDropout())

            #self._first_part_relu1.append(pelu(parameters_dimensions=(1, 8, 1, 1)))
            #self._first_part_relu2.append(pelu(parameters_dimensions=(1, 12, 1, 1)))
            self._first_part_relu1.append(nn.PReLU(8))
            self._first_part_relu2.append(nn.PReLU(12))

            self._first_part_batchnorm1.append(nn.BatchNorm2d(8, eps=1e-4))
            self._first_part_batchnorm2.append(nn.BatchNorm2d(12, eps=1e-4))

        self._list_conv1_first_part = nn.ModuleList(self._list_conv1_first_part)
        self._list_conv2_first_part = nn.ModuleList(self._list_conv2_first_part)
        self._first_part_dropout1 = nn.ModuleList(self._first_part_dropout1)
        self._first_part_dropout2 = nn.ModuleList(self._first_part_dropout2)
        self._first_part_relu1 = nn.ModuleList(self._first_part_relu1)
        self._first_part_relu2 = nn.ModuleList(self._first_part_relu2)
        self._first_part_batchnorm1 = nn.ModuleList(self._first_part_batchnorm1)
        self._first_part_batchnorm2 = nn.ModuleList(self._first_part_batchnorm2)

        self._list_conv1_second_part = []
        self._second_part_dropout1 = []
        self._second_part_relu1 = []
        self._second_part_batchnorm = []
        for i in range(2):
            self._list_conv1_second_part.append (nn.Conv2d(12, 24, kernel_size=(3, 2)))

            self._second_part_dropout1.append(McDropout())

            #self._second_part_relu1.append(pelu(parameters_dimensions=(1, 24, 1, 1)))
            self._second_part_relu1.append(nn.PReLU(24))

            self._second_part_batchnorm.append(nn.BatchNorm2d(24, eps=1e-4))

        self._list_conv1_second_part = nn.ModuleList(self._list_conv1_second_part)
        self._second_part_dropout1 = nn.ModuleList(self._second_part_dropout1)
        self._second_part_relu1 = nn.ModuleList(self._second_part_relu1)
        self._second_part_batchnorm = nn.ModuleList(self._second_part_batchnorm)

        self._conv3 = nn.Conv2d(24, 48, kernel_size=2)
        self._batch_norm_3 = nn.BatchNorm2d(48, eps=1e-4)
        #self._prelu_3 = pelu(parameters_dimensions=(1, 48, 1, 1))
        self._prelu_3 = nn.PReLU(48)
        self._dropout3 = McDropout()

        self._fc1 = nn.Linear(48, 100)
        self._batch_norm_fc1 = nn.BatchNorm1d(100, eps=1e-4)
        #self._prelu_fc1 = pelu(parameters_dimensions=(1, 100))
        self._prelu_fc1 = nn.PReLU(100)
        self._dropout_fc1 = McDropout()

        self._fc2 = nn.Linear(100, 100)
        self._batch_norm_fc2 = nn.BatchNorm1d(100, eps=1e-4)
        #self._prelu_fc2 = pelu(parameters_dimensions=(1, 100))
        self._prelu_fc2 = nn.PReLU(100)
        self._dropout_fc2 = McDropout()

        self._output = nn.Linear(100, number_of_class)

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
                torch.nn.init.kaiming_normal(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x):

        x = self._input_prelu(self._input_batch_norm(x))

        input_1 = x[:, 0:3, :, :]
        input_2 = x[:, 3:6, :, :]
        input_3 = x[:, 6:9, :, :]
        input_4 = x[:, 9:12, :, :]

        first_branch = self.first_parallel(input_1, 0)
        second_branch = self.first_parallel(input_2, 1)
        third_branch = self.first_parallel(input_3, 2)
        fourth_branch = self.first_parallel(input_4, 3)

        first_merge_1 = first_branch+second_branch
        first_merge_2 = third_branch+fourth_branch


        second_merge = self.second_parallel(first_merge_1, 0) + self.second_parallel(first_merge_2, 1)


        after_conv = self._dropout3(self._prelu_3(self._batch_norm_3(self._conv3(second_merge))))

        flatten_tensor = after_conv.view(-1, 48)

        fc1_output = self._dropout_fc1(self._prelu_fc1(self._batch_norm_fc1(self._fc1(flatten_tensor))))

        fc2_output = self._dropout_fc2(self._prelu_fc2(self._batch_norm_fc2(self._fc2(fc1_output))))


        return nn.functional.log_softmax(self._output(fc2_output))

    def first_parallel(self, input_to_give, index):
        conv1_first_part1 = self._list_conv1_first_part[index](input_to_give)
        batch_norm1_first_part1 = self._first_part_batchnorm1[index](conv1_first_part1)
        prelu1_first_part1 = self._first_part_relu1[index](batch_norm1_first_part1)
        dropout1_first_part1 = self._first_part_dropout1[index](prelu1_first_part1)

        conv1_first_part2 = self._list_conv2_first_part[index](dropout1_first_part1)
        batch_norm1_first_part2 = self._first_part_batchnorm2[index](conv1_first_part2)
        prelu1_first_part2 = self._first_part_relu2[index](batch_norm1_first_part2)
        dropout1_first_part2 = self._first_part_dropout2[index](prelu1_first_part2)

        return dropout1_first_part2

    def second_parallel(self, input_to_give, index):
        conv1_second_part = self._list_conv1_second_part[index](input_to_give)
        batch_norm1_second_part = self._second_part_batchnorm[index](conv1_second_part)
        prelu1_second_part = self._second_part_relu1[index](batch_norm1_second_part)
        dropout1_second_part = self._second_part_dropout1[index](prelu1_second_part)

        return dropout1_second_part
