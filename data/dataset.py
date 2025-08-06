from turtle import mode
import torch
from torchvision import datasets, transforms

def get_train_loader(batch_size, data_path):
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Rotation
        transforms.RandomAffine(0, shear=10),  # Shearing
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Shifting up and down
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Zooming
        transforms.Resize((28, 28)),  # Rescale
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # # Debug: Print a batch of data
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    # print(f'Train images batch shape: {images.shape}')
    # print(f'Train labels batch shape: {labels.shape}')
    
    return train_loader

def get_mnist_loaders(batch_size, data_path, mode='test'):
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Rotation
        transforms.RandomAffine(0, shear=10),  # Shearing
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Shifting up and down
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Zooming
        transforms.Resize((28, 28)),  # Rescale
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    
    if mode == 'train':
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # # Debug: Print a batch of data
    # data_iter = iter(test_loader)
    # images, labels = next(data_iter)
    # print(f'Test images batch shape: {images.shape}')
    # print(f'Test labels batch shape: {labels.shape}')
    
    return train_loader, test_loader

def get_inverted_mnist_test_loader(batch_size, data_path):
    transform = transforms.Compose([
        transforms.RandomInvert(p=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # Debug: Print a batch of data
    # data_iter = iter(test_loader)
    # images, labels = next(data_iter)
    # print(f'Test images batch shape: {images.shape}')
    # print(f'Test labels batch shape: {labels.shape}')
    
    return test_loader

def get_usps_test_loader(batch_size, data_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Rescale to 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    usps_test_dataset = datasets.USPS(data_path, train=False, download=True, transform=transform)
    usps_test_loader = torch.utils.data.DataLoader(usps_test_dataset, batch_size=batch_size, shuffle=False)
    
    return usps_test_loader