import torch
import torch.nn as nn
class SimpleCNN(nn.Module):

    def __init__(self, num_classes, img_size, dropout_p: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
        )

        feat_h = img_size // 8
        feat_w = img_size // 8
        feat_size = 64 * feat_h * feat_w

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class SimpleCNN_BN(nn.Module):
    def __init__(self, num_classes, img_size, dropout_p: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        feat_h = img_size // 8
        feat_w = img_size // 8
        feat_size = 64 * feat_h * feat_w

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class DeeperCNN(nn.Module):
    def __init__(self, num_classes, img_size):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        feat_h = img_size // 8
        feat_w = img_size // 8
        feat_size = 64 * feat_h * feat_w

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

from torchvision import models
import torch.nn as nn

class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.net = models.resnet50(weights=weights)

        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)

    def freeze_backbone(self):
        for p in self.net.parameters():
            p.requires_grad = False
        for p in self.net.fc.parameters():
            p.requires_grad = True

    def unfreeze_layer4_and_fc(self):
        # open layer4 + fc (common fine-tune setup)
        for p in self.net.layer4.parameters():
            p.requires_grad = True
        for p in self.net.fc.parameters():
            p.requires_grad = True
