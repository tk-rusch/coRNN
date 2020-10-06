[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coupled-oscillatory-recurrent-neural-network/sequential-image-classification-on-sequential)](https://paperswithcode.com/sota/sequential-image-classification-on-sequential?p=coupled-oscillatory-recurrent-neural-network)
# Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies
This repository contains the implementation to reproduce the numerical experiments 
of the paper [Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies](https://arxiv.org/pdf/2010.00951.pdf)



## Requirements

```bash
Python 3.6.1
pytorch 1.3.1
torchvision 0.4.2
torchtext 0.6.0
numpy 1.17.4
spacy v2.2+
```
If you want to run the experiments on a GPU, please make sure you have installed the corresponding cuda packages.


## Datasets

This repository contains the codes to reproduce the results of the following experiments for the proposed coRNN:

  - **The Adding Problem** 
  - **Sequential MNIST** 
  - **Permuted Sequential MNIST** 
  - **Noisy CIFAR10**
  - **HAR-2**
  - **IMDB**

The data sets for the MNIST/CIFAR10 task and the IMDB task are getting downloaded through torchvision and torchtext, respectively. 
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
    <td> 97.34% test accuarcy </td>
  </tr>
    <tr>
    <td>Noisy CIFAR10</td>
    <td> 58.2% test accuracy </td>
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



