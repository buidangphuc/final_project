from pathlib import Path
from typing import Optional, Tuple

import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class ImagesFolderDataset(Dataset):
    """Dataset for arbitrary folders of RGB images."""

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None, exts=None):
        self.root = Path(root_dir)
        extensions = exts or (".png", ".jpg", ".jpeg")
        self.files = sorted(p for p in self.root.glob("*") if p.suffix.lower() in extensions)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        img_path = self.files[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, -1, str(img_path.name)


def get_cifar_loaders(batch_size: int = 128, workers: int = 4, no_augment: bool = False) -> Tuple[DataLoader, DataLoader]:
    # CIFAR-10 normalization (standard mean and std)
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    
    transform_train = [transforms.ToTensor(), normalize]
    if not no_augment:
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    train_transform = transforms.Compose(transform_train)
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, test_loader
