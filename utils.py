import torch

from tqdm import tqdm


def train(model, device, train_loader, epoch):
    model.train()
    avg_loss = []
    print(next(model.parameters())[0])
    for i, (data, target) in enumerate(tqdm(train_loader)):
        model.train()
        data = data.to(device)
        target = target.to(device)
        loss = model.train_step(data, target)
        loss.backward()
        model.optim.step()
        model.weight_clipping()
        model.optim.zero_grad()
        avg_loss.append(loss.item())
    # print(next(model.parameters()))

    return sum(avg_loss) / len(avg_loss)


def accuracy(output: torch.tensor,
             target: torch.tensor) -> float:
    pred: torch.tensor = output.data.max(1, keepdim=True)[1]
    correct: int = pred.eq(target.data.view_as(pred)).sum()
    len_data: int = output.shape[0]

    return float(correct) / len_data


def valid(model, device, test_loader, epoch):
    avg_loss = []
    avg_acc = []
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            target = target.to(device)
            outputs = model.forward(data)
            loss = model.loss_fn(outputs, target)
            acc = accuracy(outputs, target)

            avg_loss.append(loss)
            avg_acc.append(acc)

    print('Valid Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}'.format(
        epoch,
        loss.item(),
        acc))

    return sum(avg_loss) / len(avg_loss), sum(avg_acc) / len(avg_acc)
