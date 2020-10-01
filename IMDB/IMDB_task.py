from torch import nn, optim
import torch
import model
import argparse
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=100,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument('--embedding', type=int, default=100,
                    help='embedding size for the dictonary')
parser.add_argument('--lr', type=float, default=6e-4,
                    help='learning rate')
parser.add_argument('--dt',type=float, default=5.4e-2,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma',type=float, default=4.9,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon',type=float, default = 4.8,
                    help='z controle parameter <epsilon> of the coRNN')

args = parser.parse_args()

## set up data iterators and dictonary:
train_iterator, valid_iterator, test_iterator, text_field = utils.get_data(args.batch,args.embedding)

n_inp = len(text_field.vocab)
n_out = 1
pad_idx = text_field.vocab.stoi[text_field.pad_token]

model = model.RNNModel(n_inp,args.embedding,args.n_hid,n_out,
                       pad_idx, args.dt, args.gamma, args.epsilon).to(device)

## zero embedding for <unk_token> and <padding_token>:
utils.zero_words_in_embedding(model,args.embedding,text_field,pad_idx)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def evaluate(data_iterator):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in data_iterator:
            text, text_lengths = batch.text
            predictions = model(text.to(device), text_lengths.to(device)).squeeze(1)
            loss = criterion(predictions, batch.label.to(device))
            acc = binary_accuracy(predictions, batch.label.to(device))
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(data_iterator), epoch_acc / len(data_iterator)

def train():
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in train_iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text.to(device), text_lengths.to(device)).squeeze(1)
        loss = criterion(predictions, batch.label.to(device))
        acc = binary_accuracy(predictions, batch.label.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(train_iterator), epoch_acc / len(train_iterator)

if __name__ == "__main__":
    for epoch in range(args.epochs):
        train_loss, train_acc = train()
        eval_loss, eval_acc = evaluate(valid_iterator)
        test_loss, test_acc = evaluate(test_iterator)
        print('Train set: Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(train_loss, train_acc))
        print('Valid set: Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(eval_loss, eval_acc))
        print('Test set: Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss, test_acc))


