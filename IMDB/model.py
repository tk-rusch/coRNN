import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp, n_hid)
        self.h2h = nn.Linear(n_hid+n_hid, n_hid, bias=False)

    def forward(self, x):
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        inputs = self.i2h(x)
        for t in range(x.size(0)):
            hz = hz + self.dt * (torch.tanh(self.h2h(torch.cat((hz,hy),dim=1)) + inputs[t])
                                          - self.gamma * hy - self.epsilon * hz)
            hy = hy + self.dt * hz

        return hy

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, dt, gamma, epsilon):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = coRNN(embedding_dim, hidden_dim,dt,gamma,epsilon).to(device)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        hidden = self.rnn(embedded)
        return self.readout(hidden)
