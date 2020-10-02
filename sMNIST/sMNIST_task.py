from torch import nn, optim
import torch
import network
import torch.nn.utils
import utils
from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=120,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=120,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0021,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.042,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=2.7,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=4.7,
                    help='z controle parameter <epsilon> of the coRNN')

args = parser.parse_args()
print(args)

n_inp = 1
n_out = 10
bs_test = 1000

model = network.coRNN(n_inp, args.n_hid, n_out,args.dt,args.gamma,args.epsilon)
train_loader, valid_loader, test_loader = utils.get_data(args.batch,bs_test)

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.reshape(bs_test, 1, 784)
            images = images.permute(2, 0, 1)

            output = model(images)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= i+1
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

for epoch in range(args.epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(args.batch, 1, 784)
        images = images.permute(2, 0, 1)

        optimizer.zero_grad()
        output = model(images)
        loss = objective(output, labels)
        loss.backward()
        optimizer.step()

    valid_acc = test(valid_loader)
    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/sMNIST_log.txt', 'a')
    if (epoch == 0):
        f.write('## learning rate = ' + str(args.lr) + ', dt = ' + str(args.dt) + ', gamma = ' + str(args.gamma) + ', epsilon = ' + str(args.epsilon) + '\n')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

    if (epoch+1) % 100 == 0:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

test_acc = test(test_loader)
f = open('result/sMNIST_log.txt', 'a')
f.write('final test accuracy: ' + str(round(test_acc, 2)) + '\n')
f.close()
