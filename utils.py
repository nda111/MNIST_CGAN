import torch


def label_to_onehot(labels, length=10):
    """
    :param labels: torch.Size([N, 1])
    :param length: M
    :return: torch.Size([M, M])
    """
    batch_size = labels.size(0)
    canvas = torch.zeros(batch_size, length)
    labels = labels.view(-1, 1)
    return canvas.scatter_(1, labels, 1)


def make_binary_labels(num_one, num_zero):
    ones = torch.ones(1, num_one)
    zeros = torch.zeros(1, num_zero)
    return torch.cat([ones, zeros], dim=1).view(-1, 1)


if __name__ == '__main__':
    print(label_to_onehot(torch.tensor([1, 5, 2, 3]).view(-1, 1)))
    print(make_binary_labels(3, 2))
    print(make_binary_labels(3, 0))
    print(make_binary_labels(0, 3))
