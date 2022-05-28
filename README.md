# Conditional GAN with MNIST Handwritten Digits
## Introduction
This repository is a implementation of conditional GAN
introduced by [Mehdi Mirza](https://scholar.google.com.hk/citations?hl=ko&user=c646VbAAAAAJ) and [Simon Osindero](https://scholar.google.com.hk/citations?hl=ko&user=Jq8ZS5kAAAAJ)
on [Conditional Generative Adversarial Nets, 2014](https://arxiv.org/abs/1411.1784)
with [PyTorch](https://pytorch.org).

![epoch1](https://github.com/nda111/MNIST_CGAN/blob/master/.doc/cover.png?raw=true)

## Implementation
Both generator and discriminator consists of fully-connected layers.
### Generator
> **Input**
> 
> Noise: torch.Size([batch_size, 100])
> 
> Label: torch.Size([batch_size, 10])
> 
> **Output**
> 
> torch.Size([batch_size, 1, 28, 28]) 
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.mlp(x)
        return x.view(-1, 1, 28, 28)
``` 

### Discriminator
> **Input**
> 
> Image: torch.Size([batch_size, 1, 28, 28])
> 
> Label: torch.Size([batch_size, 10])
>
> **Output**
> 
> torch.Size([batch_size, 10])
```python
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
           
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
           
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
           
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, c):
        img = img.view(-1, 28 * 28)
        x = torch.cat([img, c], dim=1)
        x = self.mlp(x)
        return x
```

## Experiments
### Dataset
Training set of MNIST Handwritten digit dataset with 60,000 image-label samples.
### Optimization
| Method        | Name   |
|---------------|--------|
| Batch size    | 128    |
| Optimizer     | Adam   |
| Learning rate | 1.0E-4 |
| LR Scheduling | n/a    |
### Training
Iteratively trained discriminator and generator in order for 100 epochs.

I applied binary cross entropy as the loss function to the tail of the discriminator.
### Result
Generated handwritten digits from 0 to 9 through 1~100 epochs.

![result](https://github.com/nda111/MNIST_CGAN/blob/master/.doc/result.gif?raw=true)
