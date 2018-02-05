import torch
import torch.nn as nn
import torch.nn.functional as F
from Pytorch_implementation.CWT.McDropout import McDropout
import numpy as np
from Pytorch_implementation.CWT import ScaleLayer


class SourceNetwork(nn.Module):
    def __init__(self, number_of_class, dropout_rate=.5):
        super(SourceNetwork, self).__init__()

        self._input_batch_norm = nn.BatchNorm2d(12, eps=1e-4)
        #self._input_prelu = pelu((1, 12, 1, 1))
        self._input_prelu = nn.PReLU(12)

        self._list_conv1_first_part = []
        self._list_conv2_first_part = []
        self._first_part_dropout1 = []
        self._first_part_dropout2 = []
        self._first_part_relu1 = []
        self._first_part_relu2 = []
        self._first_part_batch_norm1 = []
        self._first_part_batch_norm2 = []
        for i in range(4):
            self._list_conv1_first_part.append(nn.Conv2d(3, 8, kernel_size=3))
            self._list_conv2_first_part.append(nn.Conv2d(8, 12, kernel_size=3))

            self._first_part_dropout1.append(McDropout(p=dropout_rate))
            self._first_part_dropout2.append(McDropout(p=dropout_rate))

            #self._first_part_relu1.append(pelu(parameters_dimensions=(1, 8, 1, 1)))
            #self._first_part_relu2.append(pelu(parameters_dimensions=(1, 12, 1, 1)))
            self._first_part_relu1.append(nn.PReLU(8))
            self._first_part_relu2.append(nn.PReLU(12))

            self._first_part_batch_norm1.append(nn.BatchNorm2d(8, eps=1e-4))
            self._first_part_batch_norm2.append(nn.BatchNorm2d(12, eps=1e-4))

        self._list_conv1_first_part = nn.ModuleList(self._list_conv1_first_part)
        self._list_conv2_first_part = nn.ModuleList(self._list_conv2_first_part)
        self._first_part_dropout1 = nn.ModuleList(self._first_part_dropout1)
        self._first_part_dropout2 = nn.ModuleList(self._first_part_dropout2)
        self._first_part_relu1 = nn.ModuleList(self._first_part_relu1)
        self._first_part_relu2 = nn.ModuleList(self._first_part_relu2)
        self._first_part_batch_norm1 = nn.ModuleList(self._first_part_batch_norm1)
        self._first_part_batch_norm2 = nn.ModuleList(self._first_part_batch_norm2)

        self._list_conv1_second_part = []
        self._second_part_dropout1 = []
        self._second_part_relu1 = []
        self._second_part_batch_norm = []
        for i in range(2):
            self._list_conv1_second_part.append(nn.Conv2d(12, 24, kernel_size=(3, 2)))

            self._second_part_dropout1.append(McDropout(p=dropout_rate))

            #self._second_part_relu1.append(pelu(parameters_dimensions=(1, 24, 1, 1)))
            self._second_part_relu1.append(nn.PReLU(24))

            self._second_part_batch_norm.append(nn.BatchNorm2d(24, eps=1e-4))

        self._list_conv1_second_part = nn.ModuleList(self._list_conv1_second_part)
        self._second_part_dropout1 = nn.ModuleList(self._second_part_dropout1)
        self._second_part_relu1 = nn.ModuleList(self._second_part_relu1)
        self._second_part_batch_norm = nn.ModuleList(self._second_part_batch_norm)

        self._conv3 = nn.Conv2d(24, 48, kernel_size=2)
        self._batch_norm_3 = nn.BatchNorm2d(48, eps=1e-4)
        #self._prelu_3 = pelu(parameters_dimensions=(1, 48, 1, 1))
        self._prelu_3 = nn.PReLU(48)
        self._dropout3 = McDropout(p=dropout_rate)

        self._fc1 = nn.Linear(48, 100)
        self._batch_norm_fc1 = nn.BatchNorm2d(100, eps=1e-4)
        #self._prelu_fc1 = pelu(parameters_dimensions=(1, 100))
        self._prelu_fc1 = nn.PReLU(100)
        self._dropout_fc1 = McDropout(p=dropout_rate)

        self._fc2 = nn.Linear(100, 100)
        self._batch_norm_fc2 = nn.BatchNorm2d(100, eps=1e-4)
        #self._prelu_fc2 = pelu(parameters_dimensions=(1, 100))
        self._prelu_fc2 = nn.PReLU(100)
        self._dropout_fc2 = McDropout(p=dropout_rate)

        self._output = nn.Linear(100, number_of_class)

        self.initialize_weights()

        print("Number Parameters: ", self.get_n_params())

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
        batch_norm1_first_part1 = self._first_part_batch_norm1[index](conv1_first_part1)
        prelu1_first_part1 = self._first_part_relu1[index](batch_norm1_first_part1)
        dropout1_first_part1 = self._first_part_dropout1[index](prelu1_first_part1)

        conv1_first_part2 = self._list_conv2_first_part[index](dropout1_first_part1)
        batch_norm1_first_part2 = self._first_part_batch_norm2[index](conv1_first_part2)
        prelu1_first_part2 = self._first_part_relu2[index](batch_norm1_first_part2)
        dropout1_first_part2 = self._first_part_dropout2[index](prelu1_first_part2)

        return dropout1_first_part2

    def second_parallel(self, input_to_give, index):
        conv1_second_part = self._list_conv1_second_part[index](input_to_give)
        batch_norm1_second_part = self._second_part_batch_norm[index](conv1_second_part)
        prelu1_second_part = self._second_part_relu1[index](batch_norm1_second_part)
        dropout1_second_part = self._second_part_dropout1[index](prelu1_second_part)

        return dropout1_second_part

class TargetNetwork(nn.Module):
    def __init__(self, number_of_class, weights_pre_trained_cnn, dropout=.5):
        super(TargetNetwork, self).__init__()
        self._input_target_batch_norm = nn.BatchNorm2d(12, eps=1e-4)
        #self._input_prelu = pelu((1, 12, 1, 1))
        self._input_target_prelu = nn.PReLU(12)

        self._list_target_conv1_first_part = []
        self._list_target_conv2_first_part = []
        self._first_part_target_dropout1 = []
        self._first_part_target_dropout2 = []
        self._first_part_target_relu1 = []
        self._first_part_target_relu2 = []
        self._first_part_target_batch_norm1 = []
        self._first_part_target_batch_norm2 = []
        self._source_weight_learnable = []
        for i in range(4):
            self._list_target_conv1_first_part.append(nn.Conv2d(3, 8, kernel_size=3))
            self._list_target_conv2_first_part.append(nn.Conv2d(8, 12, kernel_size=3))

            self._first_part_target_dropout1.append(McDropout())
            self._first_part_target_dropout2.append(McDropout())

            #self._first_part_relu1.append(pelu(parameters_dimensions=(1, 8, 1, 1)))
            #self._first_part_relu2.append(pelu(parameters_dimensions=(1, 12, 1, 1)))
            self._first_part_target_relu1.append(nn.PReLU(8))
            self._first_part_target_relu2.append(nn.PReLU(12))

            self._first_part_target_batch_norm1.append(nn.BatchNorm2d(8, eps=1e-4))
            self._first_part_target_batch_norm2.append(nn.BatchNorm2d(12, eps=1e-4))

            self._source_weight_learnable.append(ScaleLayer.ScaleLayer((1, 8, 1, 1)))

        self._list_target_conv1_first_part = nn.ModuleList(self._list_target_conv1_first_part)
        self._list_target_conv2_first_part = nn.ModuleList(self._list_target_conv2_first_part)
        self._first_part_target_dropout1 = nn.ModuleList(self._first_part_target_dropout1)
        self._first_part_target_dropout2 = nn.ModuleList(self._first_part_target_dropout2)
        self._first_part_target_relu1 = nn.ModuleList(self._first_part_target_relu1)
        self._first_part_target_relu2 = nn.ModuleList(self._first_part_target_relu2)
        self._first_part_target_batch_norm1 = nn.ModuleList(self._first_part_target_batch_norm1)
        self._first_part_target_batch_norm2 = nn.ModuleList(self._first_part_target_batch_norm2)
        self._source_weight_learnable = nn.ModuleList(self._source_weight_learnable)

        self._list_target_conv1_second_part = []
        self._second_part_target_dropout1 = []
        self._second_part_target_relu1 = []
        self._second_part_target_batch_norm = []
        for i in range(2):
            self._list_target_conv1_second_part.append(nn.Conv2d(12, 24, kernel_size=(3, 2)))

            self._second_part_target_dropout1.append(McDropout())

            #self._second_part_relu1.append(pelu(parameters_dimensions=(1, 24, 1, 1)))
            self._second_part_target_relu1.append(nn.PReLU(24))

            self._second_part_target_batch_norm.append(nn.BatchNorm2d(24, eps=1e-4))

        self._list_target_conv1_second_part = nn.ModuleList(self._list_target_conv1_second_part)
        self._second_part_target_dropout1 = nn.ModuleList(self._second_part_target_dropout1)
        self._second_part_target_relu1 = nn.ModuleList(self._second_part_target_relu1)
        self._second_part_target_batch_norm = nn.ModuleList(self._second_part_target_batch_norm)

        self._target_conv3 = nn.Conv2d(24, 48, kernel_size=2)
        self._target_batch_norm_3 = nn.BatchNorm2d(48, eps=1e-4)
        #self._prelu_3 = pelu(parameters_dimensions=(1, 48, 1, 1))
        self._target_prelu_3 = nn.PReLU(48)
        self._target_dropout3 = McDropout()

        self._target_fc1 = nn.Linear(48, 100)
        self._target_batch_norm_fc1 = nn.BatchNorm2d(100, eps=1e-4)
        #self._prelu_fc1 = pelu(parameters_dimensions=(1, 100))
        self._target_prelu_fc1 = nn.PReLU(100)
        self._target_dropout_fc1 = McDropout()

        self._target_fc2 = nn.Linear(100, 100)
        self._target_batch_norm_fc2 = nn.BatchNorm2d(100, eps=1e-4)
        #self._prelu_fc2 = pelu(parameters_dimensions=(1, 100))
        self._target_prelu_fc2 = nn.PReLU(100)
        self._target_dropout_fc2 = McDropout()

        self._target_output = nn.Linear(100, number_of_class)

        # Define the learnable scalar that are employed to hand the importance of the source network for the target
        # network
        self._source_weight_merge_1 = ScaleLayer.ScaleLayer((1, 12, 1, 1))
        self._source_weight_merge_2 = ScaleLayer.ScaleLayer((1, 12, 1, 1))
        self._source_weight_merge_3 = ScaleLayer.ScaleLayer((1, 24, 1, 1))
        self._source_weight_merge_4 = ScaleLayer.ScaleLayer((1, 48, 1, 1))
        self._source_weight_merge_5 = ScaleLayer.ScaleLayer((1, 100))
        self._source_weight_merge_6 = ScaleLayer.ScaleLayer((1, 100))

        self.initialize_weights()

        # Start with the pre-trained model
        pre_trained_model = SourceNetwork(number_of_class=number_of_class)
        self._added_source_network_to_graph = nn.Sequential(*list(pre_trained_model.children()))

        print("Number Parameters: ", self.get_n_params())



        # Load the pre-trained model weights (Source Network)
        pre_trained_model.load_state_dict(weights_pre_trained_cnn)

        # Freeze the weights of the pre-trained model so they do not change during training of the target network
        # (except for the BN layers that will be trained as normal).
        print("Output :", pre_trained_model._list_conv2_first_part[0])
        for child in pre_trained_model.children():
            if isinstance(child, nn.ModuleList):  # We have to go one step deeper to get to the actual modules
                for module in child:
                    if isinstance(module, McDropout):
                        module.update_p(dropout)
                    elif isinstance(module, nn.BatchNorm2d) is False:
                        print(module)
                        for param in module.parameters():
                            param.requires_grad = False
            else:
                if isinstance(child, McDropout):
                    child.update_p(dropout)
                elif isinstance(child, nn.BatchNorm2d) is False:
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

        x_target = self._input_target_prelu(self._input_target_batch_norm(x))
        x_source = self._source_network["_input_prelu"](self._source_network["_input_batch_norm"](x))

        input_1_target = x_target[:, 0:3, :, :]
        input_2_target = x_target[:, 3:6, :, :]
        input_3_target = x_target[:, 6:9, :, :]
        input_4_target = x_target[:, 9:12, :, :]

        input_1_source = x_source[:, 0:3, :, :]
        input_2_source = x_source[:, 3:6, :, :]
        input_3_source = x_source[:, 6:9, :, :]
        input_4_source = x_source[:, 9:12, :, :]

        first_branch, dropout_source_first_branch = self.first_parallel(input_1_target, input_1_source, 0)
        second_branch, dropout_source_second_branch = self.first_parallel(input_2_target, input_2_source, 1)
        third_branch, dropout_source_third_branch = self.first_parallel(input_3_target, input_3_source, 2)
        fourth_branch, dropout_source_fourth_branch = self.first_parallel(input_4_target, input_4_source, 3)

        first_merge_source_1 = dropout_source_first_branch + dropout_source_second_branch
        first_merge_source_2 = dropout_source_third_branch + dropout_source_fourth_branch

        first_merge_1 = first_branch + second_branch + self._source_weight_merge_1(first_merge_source_1)
        first_merge_2 = third_branch + fourth_branch + self._source_weight_merge_2(first_merge_source_2)

        first_branch_2, dropout2_source_first_branch = self.second_parallel(first_merge_1, first_merge_source_1, 0)
        second_branch_2, dropout2_source_second_branch = self.second_parallel(first_merge_2, first_merge_source_2, 1)

        second_merge = first_branch_2 + second_branch_2 + dropout2_source_first_branch +self._source_weight_merge_3(dropout2_source_second_branch)

        after_conv = self._target_dropout3(
            self._target_prelu_3(self._target_batch_norm_3(self._target_conv3(second_merge))))

        second_merge_source = dropout2_source_first_branch + dropout2_source_second_branch
        after_conv_source = self._source_network["_dropout3"](
            self._source_network["_prelu_3"](self._source_network["_batch_norm_3"](
                self._source_network["_conv3"](second_merge_source))))

        conv_finished = after_conv + self._source_weight_merge_4(after_conv_source)

        flatten_tensor = conv_finished.view(-1, 48)
        flatten_tensor_source = after_conv_source.view(-1, 48)

        fc1_output = self._target_dropout_fc1(
            self._target_prelu_fc1(self._target_batch_norm_fc1(self._target_fc1(flatten_tensor))))
        fc1_output_source = self._source_network["_dropout_fc1"](self._source_network["_dropout_fc1"](
            self._source_network["_dropout_fc1"](self._source_network["_fc1"](flatten_tensor_source))))

        fc1_output_added = fc1_output + self._source_weight_merge_5(fc1_output_source)

        fc2_output = self._target_dropout_fc2(
            self._target_prelu_fc2(self._target_batch_norm_fc2(self._target_fc2(fc1_output_added))))

        fc2_output_source = self._source_network["_dropout_fc2"](self._source_network["_dropout_fc2"](
            self._source_network["_dropout_fc2"](self._source_network["_fc2"](fc1_output_source))))

        fc2_output_added = fc2_output + self._source_weight_merge_6(fc2_output_source)

        return nn.functional.log_softmax(self._target_output(fc2_output_added))

    def first_parallel(self, input_to_give_target, input_to_give_source, index):
        conv1_first_part1_target = self._list_target_conv1_first_part[index](input_to_give_target)
        batch_norm1_first_part1_target = self._first_part_target_batch_norm1[index](conv1_first_part1_target)
        prelu1_first_part1_target = self._first_part_target_relu1[index](batch_norm1_first_part1_target)
        dropout1_first_part1_target = self._first_part_target_dropout1[index](prelu1_first_part1_target)

        conv1_first_part1_source = self._source_network["_list_conv1_first_part"][index](input_to_give_source)
        batch_norm1_first_part1_source = self._source_network["_first_part_batch_norm1"][index](conv1_first_part1_source)
        prelu1_first_part1_source = self._source_network["_first_part_relu1"][index](batch_norm1_first_part1_source)
        dropout1_first_part1_source = self._source_network["_first_part_dropout1"][index](prelu1_first_part1_source)

        concat_first_part = dropout1_first_part1_target + self._source_weight_learnable[index](dropout1_first_part1_source)

        conv1_first_part2_target = self._list_target_conv2_first_part[index](concat_first_part)
        batch_norm1_first_part2_target = self._first_part_target_batch_norm2[index](conv1_first_part2_target)
        prelu1_first_part2_target = self._first_part_target_relu2[index](batch_norm1_first_part2_target)
        dropout1_first_part2_target = self._first_part_target_dropout2[index](prelu1_first_part2_target)

        conv1_first_part2_source = self._source_network["_list_conv2_first_part"][index](dropout1_first_part1_source)
        batch_norm1_first_part2_source = self._source_network["_first_part_batch_norm2"][index](conv1_first_part2_source)
        prelu1_first_part2_source = self._source_network["_first_part_relu2"][index](batch_norm1_first_part2_source)
        dropout1_first_part2_source = self._source_network["_first_part_dropout2"][index](prelu1_first_part2_source)

        return dropout1_first_part2_target, dropout1_first_part2_source

    def second_parallel(self, input_to_give_target, input_to_give_source, index):
        conv1_second_part_target = self._list_target_conv1_second_part[index](input_to_give_target)
        batch_norm1_second_part_target = self._second_part_target_batch_norm[index](conv1_second_part_target)
        prelu1_second_part_target = self._second_part_target_relu1[index](batch_norm1_second_part_target)
        dropout1_second_part_target = self._second_part_target_dropout1[index](prelu1_second_part_target)

        conv1_second_part_source = self._source_network["_list_conv1_second_part"][index](input_to_give_source)
        batch_norm1_second_part_source = self._source_network["_second_part_batch_norm"][index](conv1_second_part_source)
        prelu1_second_part_source = self._source_network["_second_part_relu1"][index](batch_norm1_second_part_source)
        dropout1_second_part_source = self._source_network["_second_part_dropout1"][index](prelu1_second_part_source)

        return dropout1_second_part_target, dropout1_second_part_source

'''
from torch.autograd import Variable
cnn = TargetNetwork(7)
inputs = Variable(torch.randn(1, 12, 8, 7))
cnn.eval()
h = cnn(inputs)
'''