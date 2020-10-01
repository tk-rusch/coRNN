from torch import nn
import torch
from torch.autograd import Variable
import math
from torch.nn import init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp, n_hid,bias=True)
        self.h2h1 = nn.Linear(n_hid,n_hid,bias=False)
        self.h2h2 = nn.Linear(n_hid,n_hid,bias=False)

    def forward(self,x,hy,hz):
        hz = hz + self.dt * (torch.tanh(self.h2h1(hy) + self.h2h2(hz) + self.i2h(x))
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp,n_hid,dt,gamma,epsilon)
        self.readout = nn.Linear(n_hid, n_out)

    def forward(self, x):
        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)

        for t in range(x.size(0)):
            hy, hz = self.cell(x[t],hy,hz)
        output = self.readout(hy)

        return output
