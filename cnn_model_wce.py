"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn
import constants as c
import torch

DROP_OUT = 0.5
DIMENSION = 512 * 300


class Convolutional_Neural_Network(nn.Module):

    def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
        return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2

    def __init__(self):

        super().__init__()

        #in the origine code there was 2, c.BATCH_SIZE in the nn.Conv2d function
        self.conv_2d_1 = nn.Conv2d(1, c.BATCH_SIZE, kernel_size=(3, 3), stride=(2, 2))
        self.bn_1 = nn.BatchNorm2d(c.BATCH_SIZE)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3))

        self.conv_2d_2 = nn.Conv2d(c.BATCH_SIZE, 256, kernel_size=(3, 3), stride=(2, 2))
        self.bn_2 = nn.BatchNorm2d(256)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_2d_3 = nn.Conv2d(256, 384, kernel_size=(2, 2), stride=(1, 1))
        self.bn_3 = nn.BatchNorm2d(384)

        self.conv_2d_4 = nn.Conv2d(384, 256, kernel_size=(2, 2), stride=(1, 1))
        self.bn_4 = nn.BatchNorm2d(256)

        self.conv_2d_5 = nn.Conv2d(256, 1024, kernel_size=(2, 2), stride=(1, 1))
        self.drop_1 = nn.Dropout(p=DROP_OUT)

        self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_1 = nn.Linear(1024, 4096)
        self.drop_2 = nn.Dropout(p=DROP_OUT)

        self.dense_2 = nn.Linear(4096, c.NUM_OF_SPEAKERS)  # ** Change the right side to the exact number of speakers**

        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, X):
        x = nn.ReLU()(self.conv_2d_1(X))
        x = self.bn_1(x)
        x = self.max_pool_2d_1(x)

        x = nn.ReLU()(self.conv_2d_2(x))
        x = self.bn_2(x)
        x = self.max_pool_2d_2(x)

        x = nn.ReLU()(self.conv_2d_3(x))
        x = self.bn_3(x)

        x = nn.ReLU()(self.conv_2d_4(x))
        x = self.bn_4(x)

        x = nn.ReLU()(self.conv_2d_5(x))
        x = self.drop_1(x)
        x = self.global_avg_pooling_2d(x)

        x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer
        x = nn.ReLU()(self.dense_1(x))
        x = self.drop_2(x)

        x = self.dense_2(x)

        y = self.activation(x)
        return y

    # def get_epochs(self):
    #     return 40

    # def get_learning_rate(self):
    #     return 0.0001

    # def get_batch_size(self):
    #     return 16

    def to_string(self):
        return "Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_"