import torch
from torchvision import datasets, transforms

def get_mnist_data(batch_size=64, train=False):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)