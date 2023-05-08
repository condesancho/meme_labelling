import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

from datasets import load_dataset
from transformers import ViTImageProcessor

from vit_torch_classes import CustomVitModel, TestDataSet

import os
import pandas as pd

MODEL_DIR = "../models/vit_pretrained/"
PATH = "../models/vit_pretrained/model.pth"
DATA_DIR = "../data/images"
MODEL_NAME = "google/vit-base-patch16-224-in21k"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained model
trained_model = CustomVitModel()
trained_model.load_state_dict(torch.load(PATH))
trained_model.to(device)
trained_model.eval()

# Load the testing set
test_ds = load_dataset("imagefolder", data_dir=DATA_DIR)

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# Data preprocessing
normalize = Normalize(mean=image_mean, std=image_std)
_test_transforms = Compose(
    [
        Resize((size, size)),
        ToTensor(),
        normalize,
    ]
)


def test_transforms(example):
    example = _test_transforms(example.convert("RGB"))
    return example


# Find the candidate unlabelled images
def find_candidate_images(images_path):
    """
    Finds all candidate images in the given folder and its sub-folders.

    Returns:
        images: a list of absolute paths to the discovered images.
    """
    images = []
    for root, _, files in os.walk(images_path):
        for name in files:
            file_path = os.path.abspath(os.path.join(root, name))
            if (os.path.splitext(name)[1]).lower() in [".jpg", ".png", ".jpeg"]:
                images.append(file_path)
    return images


image_list = find_candidate_images(DATA_DIR)

test_ds = TestDataSet(image_list, test_transforms)
test_dataloader = DataLoader(test_ds, batch_size=1)

# analyze images and grab predictions
predictions = []
unlabelled_imgs = []
for idx, image in enumerate(test_dataloader):
    image_path = test_dataloader.dataset.imagepath
    image = image.to(device)
    if image is not None:
        if idx % 100 == 0:
            print(
                "getting prediction for %s %d/%d"
                % (image_path, idx, len(test_dataloader))
            )

        with torch.no_grad():
            y_prob = trained_model(image)

        _, pred = torch.max(y_prob, 1)
        predictions.append(pred.item() + 1)
        unlabelled_imgs.append(image_path)

df = pd.DataFrame()
df["image"] = unlabelled_imgs
df["predicted_labels"] = predictions
df.to_csv(MODEL_DIR + "predicted_labels.csv", index=False)
