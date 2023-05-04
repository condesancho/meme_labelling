from torch import nn
from torch.utils.data import Dataset

from PIL import Image

from transformers import ViTModel

""" A custom ViT model using a pretrained backbone and a new head """


class CustomVitModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomVitModel, self).__init__()

        self.base_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # output features from vit is 768
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )

    def forward(self, input):
        outputs = self.base_model(input)
        # The new head
        outputs = self.head(outputs[1])

        return outputs


""" Early Stopping class """


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.max_val_acc = 0.0

    def __call__(self, current_val_acc):
        if (current_val_acc - self.max_val_acc) < 0:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.max_val_acc = current_val_acc
            self.counter = 0


""" The testing dataset class"""


class TestDataSet(Dataset):
    def __init__(self, imagelist, transform):
        self.imagepaths = imagelist
        self.transform = transform

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, index):
        self.imagepath = self.imagepaths[index]
        self.image = Image.open(self.imagepath)

        self.i = self.transform(self.image)
        return self.i
