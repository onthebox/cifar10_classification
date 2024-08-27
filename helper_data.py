import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_dataloaders_cifar10(batch_size: int, num_workers: int = 0,
                            val_fraction: float | None = None,
                            train_transforms: torchvision.transforms.transforms.Compose | None = None,
                            test_transforms: torchvision.transforms.transforms.Compose | None = None):
    """_summary_

    Args:
        batch_size (int): _description_
        num_workers (int, optional): _description_. Defaults to 0.
        val_fraction (float | None, optional): _description_. Defaults to None.
        train_transforms (torchvision.transforms.transforms.Compose | None, optional): _description_. Defaults to None.
        test_transforms (torchvision.transforms.transforms.Compose | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if not train_transforms:
        train_transforms = transforms.ToTensor()

    if not test_transforms:
        test_transforms = transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(
        root='cifar10_data',
        train=True,
        transform=train_transforms,
        download=True
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root='cifar10_data',
        train=True,
        transform=test_transforms,
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='cifar10_data',
        train=False,
        transform=test_transforms,
        download=True
    )

    if val_fraction:
        n = int(val_fraction * 50000)
        train_indices = torch.arange(0, 50000 - n)
        val_indices = torch.arange(50000 - n, 50000)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=True
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            drop_last=True
        )

    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True
        )

    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True
    )

    if not val_fraction:
        return train_loader, test_loader
    else:
        return train_loader, val_loader, test_loader
