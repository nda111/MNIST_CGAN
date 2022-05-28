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


if __name__ == '__main__':
    z = torch.randn(32, 100)
    c = torch.zeros(32, 10)

    g = Generator()
    fake = g(z, c)
    print(fake.shape)

    d = Discriminator()
    y = d(fake, c)
    print(y.shape)
