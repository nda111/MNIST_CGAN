import os
import pathlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from models import Generator, Discriminator
from utils import label_to_onehot, make_binary_labels

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

save_path = 'output'
path = pathlib.Path(save_path)
path.mkdir(parents=True, exist_ok=True)

# Dataset, DataLoader
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = MNIST('.', train=True, transform=input_transform, download=True)
test_dataset = MNIST('.', train=False, transform=input_transform, download=True)

batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Model, Optimizer
G = Generator().to(device)
D = Discriminator().to(device)
optim_G = optim.Adam(G.parameters(), lr=1.0E-4, betas=(0.9, 0.999))
optim_D = optim.Adam(D.parameters(), lr=1.0E-4, betas=(0.9, 0.999))

# Epochs
num_epochs = 100
sample_noise = torch.randn(10, 100).to(device)
for epoch in range(1, num_epochs + 1):
    print('epoch', epoch)
    # Train
    for img, label in tqdm(train_dataloader):
        real = img.to(device)
        onehot = label_to_onehot(label).to(device)
        batch_size = real.size(0)

        # Discriminator
        D.train()
        G.eval()
        z = torch.randn(batch_size, 100).to(device)
        fake = G(z, onehot).detach()

        fake_out = D(fake, onehot)
        real_out = D(real, onehot)

        fake_y = make_binary_labels(0, batch_size).to(device)
        real_y = make_binary_labels(batch_size, 0).to(device)

        fake_loss = F.binary_cross_entropy(fake_out, fake_y)
        real_loss = F.binary_cross_entropy(real_out, real_y)

        loss = fake_loss + real_loss
        optim_D.zero_grad()
        loss.backward()
        optim_D.step()

        # Generator
        G.train()
        D.eval()
        z = torch.randn(batch_size, 100).to(device)
        fake = G(z, onehot)
        fake_out = D(fake, onehot)
        y = make_binary_labels(batch_size, 0).to(device)

        loss = F.binary_cross_entropy(fake_out, y)
        optim_G.zero_grad()
        loss.backward()
        optim_G.step()

    # Test
    D.eval()
    G.eval()

    labels = torch.arange(10).long()
    onehot = label_to_onehot(labels)

    fake = G(sample_noise, onehot)
    for i in range(10):
        label = labels[i]
        img = fake[i][0].detach()
        plt.subplot(1, 10, i + 1)
        plt.imshow(img, cmap='gray')
    plt.savefig(os.path.join(save_path, f'%03d.png' % epoch), dpi=300)

torch.save({
    'G': G.state_dict(),
    'D': D.state_dict(),
}, os.path.join(save_path, 'state_dict.pkl'))
