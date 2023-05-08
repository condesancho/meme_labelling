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

from vit_torch_classes import (
    CustomVitModel,
    TestDataSet,
    FeatureExtractor,
)
import os
import numpy as np
import pandas as pd

MODEL_DIR = "../models/vit_feat_ex/"
PATH = "../models/vit_pretrained/model.pth"
LABELLED_DATA_DIR = "../data/torch"
UNLABELLED_DATA_DIR = "../data/images"
MODEL_NAME = "google/vit-base-patch16-224-in21k"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained model
trained_model = CustomVitModel()
trained_model.load_state_dict(torch.load(PATH))
trained_model.eval()

feat_ex_model = FeatureExtractor(trained_model, ["base_model"])
feat_ex_model.to(device)

# Load the training set
test_ds = load_dataset("imagefolder", data_dir=UNLABELLED_DATA_DIR)

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


def train_transforms(examples):
    examples["pixel_values"] = [
        _test_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


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


""" Feature extraction for the labelled images """
train_ds = load_dataset("imagefolder", data_dir=LABELLED_DATA_DIR, split="train")
train_ds.set_transform(train_transforms)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


train_dataloader = torch.utils.data.DataLoader(
    train_ds, batch_size=1, collate_fn=collate_fn
)
activations = []
labels = []

for idx, image in enumerate(train_dataloader):
    image["pixel_values"] = image["pixel_values"].to(device)
    if image is not None:
        with torch.no_grad():
            features = feat_ex_model(image["pixel_values"])
            output = features["base_model"]
            activations.append(output.view(-1).cpu().numpy())
            labels.append(image["labels"].numpy())
    break

labelled_features = np.array(activations)

df_labelled = pd.DataFrame(data=labelled_features)
labels = np.array(labels) + 1
df_labelled["label"] = labels
df_labelled.to_csv(MODEL_DIR + "labelled_features.csv", index=False)

""" Feature extraction for the unlabelled images """
unlabelled_image_list = find_candidate_images(UNLABELLED_DATA_DIR)

test_ds = TestDataSet(unlabelled_image_list, test_transforms)
test_dataloader = DataLoader(test_ds, batch_size=1)
activations = []
unlabelled_imgs = []

for idx, image in enumerate(test_dataloader):
    image_path = test_dataloader.dataset.imagepath
    image = image.to(device)
    if image is not None:
        if idx % 100 == 0:
            print(
                "getting features for %s %d/%d"
                % (image_path, idx, len(test_dataloader))
            )
        with torch.no_grad():
            features = feat_ex_model(image)
            output = features["base_model"]
            activations.append(output.view(-1).cpu().numpy())
            unlabelled_imgs.append(image_path)

unlabelled_features = np.array(activations)

df_unlabelled = pd.DataFrame(data=unlabelled_features)
df_unlabelled["image"] = unlabelled_imgs
df_unlabelled.to_csv(MODEL_DIR + "unlabelled_features.csv", index=False)
