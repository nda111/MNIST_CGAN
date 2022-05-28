import torch
import torch.nn as nn


def label_encoder():
    return nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.BatchNorm1d(16),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
    )


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024 - 32),
            nn.ReLU(),
            nn.BatchNorm1d(1024 - 32),
        )
        self.mlp2 = label_encoder()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=(3, 3)),
            nn.Tanh(),
        )

    def forward(self, z, c):
        """
        :param z: torch.Size([N, 100])
        :param c: torch.Size([N, 10])
        :return: torch.Size([N, 256, 256])
        """
        z = self.mlp1(z)
        c = self.mlp2(c)
        x = torch.cat([z, c], dim=1).view(-1, 1, 32, 32)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), dilation=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=(3, 3), dilation=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=(3, 3), dilation=(2, 2)),
        )
        self.mlp1 = label_encoder()
        self.mlp2 = nn.Sequential(
            nn.Linear(288, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, c):
        x = self.conv(x)
        c = self.mlp1(c)
        x = x.view(-1, 256)
        x = torch.cat([x, c], dim=1)
        x = self.mlp2(x)
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
