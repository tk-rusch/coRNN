from torch import nn, optim, Tensor
import torch
import network
import numpy as np
import torch.nn.utils
import argparse
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=64,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=250,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.017,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.1,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=0.2,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=6.4,
                    help='z controle parameter <epsilon> of the coRNN')

args = parser.parse_args()
print(args)

## Define the RNN model:
n_input = 9
n_output = 1
model = network.coRNN(n_input, args.n_hid, n_output,args.dt,args.gamma,args.epsilon)

## Define data loader:
train_data, train_labels = np.load('data/trainx'), np.load('data/trainy')
valid_data, valid_labels = np.load('data/evalx'), np.load('data/evaly')
test_data, test_labels = np.load('data/testx'), np.load('data/testy')

## Train data:
train_dataset = TensorDataset(Tensor(train_data).float(), Tensor(train_labels))
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch)

## Valid data:
valid_dataset = TensorDataset(Tensor(valid_data).float(), Tensor(valid_labels))
validloader = DataLoader(valid_dataset, shuffle=False, batch_size=valid_labels.size)

## Test data
test_dataset = TensorDataset(Tensor(test_data).float(), Tensor(test_labels).long())
testloader = DataLoader(test_dataset, shuffle=False, batch_size=test_labels.size)

objective = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def test(dataloader):
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):
            data = data.permute(1, 0, 2)
            predictions = model(data).squeeze(1)
            acc = binary_accuracy(predictions, labels)
            epoch_acc += acc.item()
    accuracy = epoch_acc / len(dataloader)
    return accuracy.item()

for epoch in range(args.epochs):
    model.train()
    for i, (data, labels) in enumerate(trainloader):
        data = data.permute(1,0,2)
        optimizer.zero_grad()
        output = model(data).squeeze(1)
        loss = objective(output, labels)
        loss.backward()
        optimizer.step()

    eval_acc = test(validloader)
    test_acc = test(testloader)
    print('Valid set: Accuracy: {:.2f}%\n'.format(eval_acc))
    print('Test set:  Accuracy: {:.2f}%\n'.format(test_acc))
