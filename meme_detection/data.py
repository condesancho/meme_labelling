import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from utils import split_dataset

from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, directory, split, reg_img_split, transform=None):
        self.train, self.val, self.test = split_dataset(directory, reg_img_split)
        self.transform = transform
        self.images = (
            self.train
            if split == "train"
            else self.val
            if split == "val"
            else self.test
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.images["file_path"][idx]
        image = image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.images["label"][idx]

        return image, label
