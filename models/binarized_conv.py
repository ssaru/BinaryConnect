import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary
from layers.binarized_conv2d_layer import BinarizedConv2d
from layers.binarized_linear_layer import BinarizedLinear


class Binarized_Conv(nn.Module):
    def __init__(self,
                 mode: str = "stochastic",
                 is_dropout: bool = False,
                 dropout_prob: float = 0.5,
                 optimizer: optim = optim.SGD,
                 learning_rate: float = 0.01,
                 momentum: float = 0,
                 weight_decay: float = 1e-5,
                 scheduler: optim.lr_scheduler = None,
                 scheduler_gamma: float = 0.1):
        super(Binarized_Conv, self).__init__()

        # Layers
        #self.conv1 = nn.Conv2d(1, 128, 3, bias=True)
        self.conv1 = BinarizedConv2d(1, 128, 3, bias=True, mode=mode)
        self.batch1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)

        #self.conv2 = nn.Conv2d(128, 256, 3, bias=True)
        self.conv2 = BinarizedConv2d(128, 256, 3, bias=True, mode=mode)
        self.batch2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)

        #self.conv3 = nn.Conv2d(256, 512, 3, bias=True)
        self.conv3 = BinarizedConv2d(256, 512, 3, bias=True, mode=mode)
        self.batch3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = BinarizedLinear(512, 10, bias=True, mode=mode)

        self.relu = nn.ReLU()

        # Dropout
        self.is_dropout = is_dropout
        if self.is_dropout:
            self.dropout1 = nn.Dropout(dropout_prob)

        # Optimizer
        self.optim = optimizer(self.parameters(),
                               lr=learning_rate,
                               momentum=momentum,
                               weight_decay=weight_decay)

        # Optimizer Scheduler
        if scheduler:
            self.scheduler = scheduler(self.optim, scheduler_gamma)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # print(x.shape)
        x = x.view(x.size()[0], -1)
        # print(x.shape)

        x = self.fc1(x)

        return x

    def weight_clipping(self):
        with torch.no_grad():
            for key in self.state_dict():
                if "weight" in key:
                    value = self.state_dict().get(key)
                    value.clamp_(-1, 1)

    def summary(self):
        summary(self, (1, 28, 28))

    def train_step(self, data: torch.tensor, target: torch.tensor) -> torch.tensor:
        outputs = self.forward(data)
        return self.loss_fn(outputs, target)

    def optim_step(self):
        if self.scheduler:
            self.scheduler.step()
        else:
            self.optim.step()

    def count_tp(self, outputs: torch.tensor, target: torch.tensor) -> int:
        self.eval()
        pred: torch.tensor = outputs.data.max(1, keepdim=True)[1]
        tp: int = pred.eq(target.data.view_as(pred)).sum()

        return tp
