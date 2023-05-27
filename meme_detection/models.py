import torch
import torch.nn as nn

from torchvision.models import resnet152, ResNet152_Weights

from transformers import ViTModel


class CustomResNet(torch.nn.Module):
    def __init__(self, num_trainable_layers, dropout):
        super(CustomResNet, self).__init__()

        conv_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        num_ftrs = conv_model.fc.in_features
        conv_model.fc = nn.Identity()

        # Freeze the appropriate number of layers
        num_children = 0
        for child in conv_model.children():
            num_children += 1

        num_layers_2_freeze = num_children - num_trainable_layers
        if num_layers_2_freeze < 0:
            error = ValueError(
                f"CustomResNet: number of trainable layers exceed number of ResNet layers which are: {num_children}"
            )
            raise error

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


class CustomVitModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomVitModel, self).__init__()

        self.base_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # output features from vit is 768
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        outputs = self.base_model(input)
        # The new head
        outputs = self.head(outputs[1])

        return outputs
