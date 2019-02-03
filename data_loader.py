import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np


class Flatten(object):
    def __call__(self, tensor):
        return tensor.view(-1)

    def __repr__(self):
        return self.__class__.__name__


class Transpose(object):
    def __call__(self, tensor):
        return tensor.permute(1, 2, 0)

    def __repr__(self):
        return self.__class__.__name__


class PretrainedCIFAR10(Dataset):

    def __init__(self, test=False):
        self.test = test

        self.X_train = np.load('data/cifar10_features_train.npy')
        self.y_train = np.load('data/cifar10_targets_train.npy')

        self.X_test = np.load('data/cifar10_features_test.npy')
        self.y_test = np.load('data/cifar10_targets_test.npy')

    def __len__(self):
        return len(self.X_test) if self.test else len(self.X_train)

    def __getitem__(self, idx):
        if self.test:
            x, y = self.X_test[idx], self.y_test[idx]
        else:
            x, y = self.X_train[idx], self.y_train[idx]

        return x, y


class PretrainedSVHN(Dataset):

    def __init__(self, test=False):
        self.test = test

        self.X_train = np.load('data/svhn_features_train.npy')
        self.y_train = np.load('data/svhn_targets_train.npy')

        self.X_test = np.load('data/svhn_features_test.npy')
        self.y_test = np.load('data/svhn_targets_test.npy')

    def __len__(self):
        return len(self.X_test) if self.test else len(self.X_train)

    def __getitem__(self, idx):
        if self.test:
            x, y = self.X_test[idx], self.y_test[idx]
        else:
            x, y = self.X_train[idx], self.y_train[idx]

        return x, y


def load_dataset(name):
    if name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Transpose()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Transpose()
        ])
        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
    elif name == 'cifar10_pretrained':
        trainset = PretrainedCIFAR10(test=False)
        testset = PretrainedCIFAR10(test=True)
    elif name == 'svhn':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Transpose()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Transpose()
        ])
        trainset = torchvision.datasets.SVHN(root='data', split='train', download=True, transform=train_transform)
        testset = torchvision.datasets.SVHN(root='data', split='test', download=True, transform=test_transform)
    elif name == 'svhn_pretrained':
        trainset = PretrainedSVHN(test=False)
        testset = PretrainedSVHN(test=True)
    else:
        raise ValueError("Unsupported dataset!")

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=200,
                                              shuffle=True,
                                              num_workers=1)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=1)
    return trainloader, testloader
