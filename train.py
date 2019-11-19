import hydra
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import argparse
from models.conv import CNN
from models.mlp import MLP
from models.binarized_mlp import Binarized_MLP


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for i, (data, target) in tqdm(enumerate(train_loader, 0)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))

@hydra.main()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--bnn_type', type=str, default='Stochastic')
    parser.add_argument('--dataset', type=str, default='CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = data_loader(dataset=args.dataset, batch_size=args.batch_size)[0]

    if args.model == 'CNN':
        model = CNN()
    elif args.model == 'MLP':
        model = MLP()
    elif args.model == 'BNN':
        if args.bnn_type == "Stochastic" or args.bnn_type == "Deterministic":
            model = Binarized_MLP(args.bnn_type)
        else:
            raise RuntimeError("not supported quantization method")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)

    if (args.save_model):
        torch.save(model.state_dict(), "CIFAR-10_MLP.pt")


if __name__ == "__main__":
    main()
