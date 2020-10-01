import torch
import numpy as np

def get_batch(T,batch_size):
    values = torch.rand(T, batch_size, requires_grad=False)
    indices = torch.zeros_like(values)
    half = int(T / 2)
    for i in range(batch_size):
        half_1 = np.random.randint(half)
        hals_2 = np.random.randint(half, T)
        indices[half_1, i] = 1
        indices[hals_2, i] = 1

    data = torch.stack((values, indices), dim=-1)
    targets = torch.mul(values, indices).sum(dim=0)
    return data, targets