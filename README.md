<h1 align='center'> Coupled Oscillatory Recurrent Neural Network (coRNN)<br>
    [ICLR 2021 Oral] </h1>


This repository contains the implementation to reproduce the numerical experiments 
of the *International Conference on Learning Representations (ICLR) 2021* **[oral]** paper [Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies](https://openreview.net/forum?id=F3s69XzWOia)



## Requirements

```bash
pytorch 1.3+
torchvision 0.4+
torchtext 0.6+
numpy 1.17+
spacy v2.2+
```
If you want to run the experiments on a GPU, please make sure you have installed the corresponding cuda packages.


## Example
The coRNN cell can be implemented in pytorch as easy as this: 
```python
from torch import nn
import torch

class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma=1., epsilon=1.):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self,x,hy,hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy),1)))
                                   - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz
```


## Datasets

This repository contains the codes to reproduce the results of the following experiments for the proposed coRNN:

  - **The Adding Problem** 
  - **Sequential MNIST** 
  - **Permuted Sequential MNIST** 
  - **Noise padded CIFAR-10**
  - **HAR-2**
  - **IMDB**

The data sets for the MNIST/CIFAR-10 task and the IMDB task are getting downloaded through torchvision and torchtext, respectively.
The data set for the HAR-2 has to be downloaded and preprocessed according to the instructions mentioned in the paper.

## Results
The results of the coRNN for each of the experiments are:
<table>
  <tr>
    <td> Experiment </td>
    <td> Result </td>
  </tr>
  <tr>
    <td>sMNIST </td>
    <td> 99.4% test accuracy</td>
  </tr>
  <tr>
    <td>psMNIST </td>
    <td> 97.3% test accuarcy </td>
  </tr>
    <tr>
    <td>Noise padded CIFAR-10</td>
    <td> 59.0% test accuracy </td>
  </tr>
  <tr>
    <td>HAR-2</td>
    <td> 97.2 test accuracy  </td>
  </tr>
  <tr>
    <td>IMDB</td>
    <td> 87.4% test accuracy </td>
  </tr>
</table>

## Citation
If you found this work useful, please consider citing
```bibtex
@inproceedings{rusch2021coupled,
  title={Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies},
  author={Rusch, T Konstantin and Mishra, Siddhartha},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
