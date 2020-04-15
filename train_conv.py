import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.binarized_conv import Binarized_Conv
from utils import train, valid, accuracy


# Classes
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Set Common Parameters
batch_size: int = 256
epochs: int = 1000
model_name = "./CONV_BinaryConnect.pth"

# Transform
transform = transforms.Compose([transforms.ToTensor()])

# Dataset directory
root = "./data"
MNIST_train_datasets = torchvision.datasets.MNIST(root,
                                                  train=True,
                                                  transform=transform,
                                                  target_transform=None,
                                                  download=True)

MNIST_test_datasets = torchvision.datasets.MNIST(root,
                                                 train=False,
                                                 transform=transform,
                                                 target_transform=None,
                                                 download=True)

train_loader = torch.utils.data.DataLoader(dataset=MNIST_train_datasets,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=MNIST_test_datasets,
                                          batch_size=len(MNIST_test_datasets),
                                          shuffle=False)

# TensorBoard
writer = SummaryWriter()

# device
device: torch.device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')

# Optimizer
optimizer = optim.SGD

# Exponential Learning Rate Decay
scheduler = torch.optim.lr_scheduler.ExponentialLR

scheduler_gamma = 0.1
learning_rate = 0.01
momentum = 0.
weight_decay = 0.

model = Binarized_Conv(mode="stochastic",
                       is_dropout=False,
                       dropout_prob=1.0,
                       optimizer=optimizer,
                       learning_rate=learning_rate,
                       momentum=momentum,
                       weight_decay=weight_decay,
                       scheduler=scheduler,
                       scheduler_gamma=scheduler_gamma)

# Train
model.to(device)

train_loss_per_epoch = [0]
valid_loss_per_epoch = [0]
valid_acc_per_epoch = [0]

not_improved_count = 0

for epoch in range(epochs):

    train_loss = train(model, device, train_loader, epoch)
    valid_loss, valid_acc = valid(model, device, test_loader, epoch)

    if valid_acc > max(valid_acc_per_epoch):
        not_improved_count = 0
    else:
        not_improved_count += 1
        print("ACC not improved. count : {}".format(not_improved_count))
        print("Present Acc : {}, Max Acc : {}".format(
            valid_acc, max(valid_acc_per_epoch)))

    if not_improved_count > 1000:
        print("Early Stopping")
        break

    writer.add_scalar(
        "Vanila BinaryConnect Conv stochastic/Loss/train", train_loss, epoch)
    writer.add_scalar(
        "Vanila BinaryConnect Conv stochastic/Loss/valid", valid_loss, epoch)
    writer.add_scalar(
        "Vanila BinaryConnect Conv stochastic/Acc/valid", valid_acc, epoch)
    train_loss_per_epoch.append(train_loss)
    valid_loss_per_epoch.append(valid_loss)
    valid_acc_per_epoch.append(valid_acc)

torch.save(model.state_dict(), model_name)
print("Save Model")
