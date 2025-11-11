import os
import torch
import torchvision
import torchvision.transforms as transforms
from contextlib import contextmanager
import sys
import os 

class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

    CIFAR100 = CIFAR10


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def loaders(dataset, path, batch_size, num_workers, transform_name,
            shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    with suppress_stdout():
        train_set = ds(path, train=True, download=True, transform=transform.train)
        train_set.data = train_set.data[:-5000]
        train_set.targets = train_set.targets[:-5000]

    print('You are going to run models on the test set. Are you sure?')
    with suppress_stdout():
        test_set = ds(path, train=False, download=True, transform=transform.test)
    val_set = ds(path, train=True, download=True, transform=transform.test)
    val_set.train = False
    val_set.data = train_set.data[-5000:]
    val_set.targets = train_set.targets[-5000:]


    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'validation': torch.utils.data.DataLoader(
                   val_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, max(train_set.targets) + 1
