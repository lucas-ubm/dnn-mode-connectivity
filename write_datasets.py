import os
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import torchvision

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Download CIFAR-10 dataset
    datasets = {
        'train': torchvision.datasets.CIFAR10('./data', train=True, download=True),
        'test': torchvision.datasets.CIFAR10('./data', train=False, download=True)
    }

    # Write datasets
    for (name, ds) in datasets.items():
        path = os.path.join('./data', f'cifar_{name}.ffcv')
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)
        print(f"Created {path}")

if __name__ == '__main__':
    main() 