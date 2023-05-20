import torch
import torch.nn as nn

from torchvision.models import resnet152, vgg16, efficientnet_b4

import numpy as np


class CustomCNN(nn.Module):
    def __init__(self, depth, dropout=0.5, hidden_1=1024, hidden_2=512):
        super(CustomCNN, self).__init__()

        # Check if the depth input is valid
        valid_depth = [4, 5, 6]
        if depth not in valid_depth:
            raise ValueError(f"CustomCNN: depth must be one of {valid_depth}")
        elif depth == 4:
            filters = [(64, 2), (128, 1), (128, 1)]
            conv_output = 128 * 14 * 14
        elif depth == 5:
            filters = [(64, 1), (128, 2), (256, 1), (256, 1)]
            conv_output = 256 * 7 * 7
        elif depth == 6:
            filters = [(64, 1), (128, 2), (256, 2), (256, 1)]
            conv_output = 256 * 7 * 7

        conv_modules = []
        conv_modules.append(nn.Conv2d(3, 64, kernel_size=3, padding="same"))
        conv_modules.append(nn.ReLU())
        conv_modules.append(nn.BatchNorm2d(64))
        conv_modules.append(nn.MaxPool2d(kernel_size=2))

        channel_output = 64
        for n_filters, n_convs in filters:
            for _ in np.arange(n_convs):
                conv_modules.append(
                    nn.Conv2d(channel_output, n_filters, kernel_size=3, padding="same")
                )
                conv_modules.append(nn.ReLU())

                # The new channel output
                channel_output = n_filters

            conv_modules.append(nn.BatchNorm2d(channel_output))
            conv_modules.append(nn.MaxPool2d(kernel_size=2))

        self.conv = nn.Sequential(*conv_modules)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x


class CustomResNet(torch.nn.Module):
    def __init__(self, num_trainable_layers, dropout):
        super(CustomResNet, self).__init__()

        conv_model = resnet152(pretrained=True)
        num_ftrs = conv_model.fc.in_features
        conv_model.fc = nn.Identity()

        # Freeze the appropriate number of layers
        num_children = 0
        for child in conv_model.children():
            num_children += 1

        num_layers_2_freeze = num_children - num_trainable_layers

        # Leave the last num_trainable_layers unfrozen to be trained
        ct = 0
        for child in conv_model.children():
            if ct < num_layers_2_freeze:
                for param in child.parameters():
                    param.requires_grad = False
            ct += 1

        self.conv_model = conv_model

        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_model(x)
        return self.fc(x)


def main():
    model = CustomResNet(5, 0.2)
    dummy_input = torch.rand((1, 3, 224, 224))
    print(model(dummy_input).size())


if __name__ == "__main__":
    main()
