import torch.nn as nn
import torchvision

class AllConv(nn.Module):
    """All-CNN-C architecture from 'Striving for Simplicity: The All Convolutional Net'.
    Used in One Pixel Attack paper for CIFAR-10.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 3, stride=1, padding=1),  # 0
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),  # 2
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=2, padding=1),  # 4: stride 2 for downsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, 3, stride=1, padding=1),  # 6
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # 8
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 10: dropout 0.3 as per paper
            nn.Conv2d(192, 192, 3, stride=2, padding=1),  # 11: stride 2 for downsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # 13
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),  # 15
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, 1),  # 17
            nn.ReLU(inplace=True),
            nn.AvgPool2d(6, stride=1),  # average pooling kernel=6, stride=1 as per paper
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)



class NiN(nn.Module):
    """Network in Network architecture for CIFAR-10.
    Used in One Pixel Attack paper.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            # mlpconv block 1
            nn.Conv2d(3, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.5),
            # mlpconv block 2
            nn.Conv2d(96, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Dropout(0.5),
            # mlpconv block 3
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8),  # global average pooling
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class VGG16CIFAR(nn.Module):
    """VGG16 adapted for CIFAR-10 (32x32 images).
    Used in One Pixel Attack paper.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(v))  # add batch normalization
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        self.features = nn.Sequential(*layers)
        # For CIFAR-10 (32x32 input), after 5 maxpools: 32/(2^5) = 1x1
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten to [batch, 512]
        return self.classifier(x)


def alexnet_bvlc(num_classes: int = 1000, pretrained: bool = True):
    """AlexNet with optional pretrained weights from torchvision.
    
    Args:
        num_classes: Number of output classes
        pretrained: If True, loads ImageNet pretrained weights (for num_classes=1000)
    """
    # Use pretrained weights if available and num_classes matches
    use_pretrained = pretrained and num_classes == 1000
    try:
        # Try new API first (torchvision >= 0.13)
        from torchvision.models import AlexNet_Weights
        weights = AlexNet_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = torchvision.models.alexnet(weights=weights)
    except ImportError:
        # Fall back to old API
        model = torchvision.models.alexnet(pretrained=use_pretrained)
    
    if num_classes != 1000:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    return model
