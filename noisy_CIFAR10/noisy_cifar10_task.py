from torch import nn, optim
import torch
import utils
import network
import argparse
import torch.nn.utils

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=250,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=100,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0075,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.034,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=1.3,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=6.4,
                    help='z controle parameter <epsilon> of the coRNN')

args = parser.parse_args()
print(args)

n_inp = 96
n_out = 10
model = network.coRNN(n_inp, args.n_hid, n_out,args.dt, args.gamma, args.epsilon)
train_loader, valid_loader, test_loader = utils.get_data(args.batch,1000)


## Define the loss
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

rands = torch.randn(1, 1000 - 32, 96)
rand_train = rands.repeat(args.batch,1,1)
rand_test = rands.repeat(1000,1,1)

def test(data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            ## Reshape images for sequence learning:
            images = torch.cat((images.permute(0,2,1,3).reshape(1000,32,96),rand_test),dim=1).permute(1,0,2)
            output = model(images)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

## train the model:
for e in range(args.epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        ## Reshape images for sequence learning:
        images = torch.cat((images.permute(0,2,1,3).reshape(args.batch,32,96),rand_train),dim=1).permute(1,0,2)
        # Training pass
        optimizer.zero_grad()
        output = model(images)
        loss = objective(output, labels)
        loss.backward()
        optimizer.step()
    eval_acc = test(valid_loader)
    test_acc = test(test_loader)
    print('Valid set: Accuracy: {:.2f}%\n'.format(eval_acc))
    print('Test set:  Accuracy: {:.2f}%\n'.format(test_acc))
