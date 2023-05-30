import torch
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from torch.utils.data import DataLoader

import numpy as np

from transformers import ViTImageProcessor

import argparse
import sys
import os
import pickle
import random
import warnings
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

from train import collate_fn
from data import ImageDataset
from models import CustomResNet, CustomVitModel

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def process_arguments(args):
    parser = argparse.ArgumentParser(description="Model fine tuning on meme detection")
    parser.add_argument(
        "--dataset_split",
        action="store",
        help="the kind of split to perform on the regular images",
        choices=["coco", "conc_capt", "icdar", "equally"],
        default="equally",
    )
    parser.add_argument(
        "--model",
        action="store",
        help="the model to be used for training",
        choices=["resnet", "vit"],
        default="resnet",
    )

    params = vars(parser.parse_args(args))
    return params


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    raise Exception("No GPUs available!")

params = process_arguments(sys.argv[1:])
dataset_split = params["dataset_split"]
model_name = params["model"]

model_dir = os.path.join("./results", model_name, dataset_split)
path = os.path.join(model_dir, "best_model.pth")
data_dir = "../data"  # directory where the folder meme_detection_dataset is stored
save_file = os.path.join(model_dir, "hyperparam_tuning.pickle")

with open(save_file, "rb") as h:
    results = pickle.load(h)

# Find the max accuracy
accuracies = []
for item in results:
    accuracies.append(item["accuracy"])
max_idx = accuracies.index(max(accuracies))

best_config = results[max_idx]["config"]

print("The best hyperparameters were:")
print(f"Learning rate:{best_config['lr']}")
print(f"Dropout:{best_config['dropout']}")

# Make the model and initialize parameters
imnet_mean = [0.485, 0.456, 0.406]
imnet_std = [0.229, 0.224, 0.225]
if model_name == "resnet":
    print(f"Trainable layers:{best_config['trainable_layers']}")
    trained_model = CustomResNet(
        best_config["trainable_layers"], best_config["dropout"]
    )
    IMG_SIZE = 224
else:
    print(f"Weight decay:{best_config['weight_decay']}")
    trained_model = CustomVitModel(best_config["dropout"])
    vit_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(vit_name)
    IMG_SIZE = processor.size["height"]

# Load trained model
trained_model.load_state_dict(torch.load(path))
trained_model.to(device)
trained_model.eval()

test_ds = ImageDataset(
    directory=data_dir,
    split="test",
    reg_img_split=dataset_split,
    transform=Compose(
        [
            Resize((IMG_SIZE, IMG_SIZE)),
            ToTensor(),
            Normalize(imnet_mean, imnet_std),
        ]
    ),
)

test_dataloader = DataLoader(
    test_ds,
    batch_size=1,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)

# analyze images and grab predictions
y_true = []
y_pred = []
for idx, item in enumerate(test_dataloader):
    inputs = item["pixel_values"].to(device)
    labels = item["labels"].to(device)
    if inputs is not None:
        if idx % 1000 == 0:
            print("getting prediction for image %d/%d" % (idx, len(test_dataloader)))

        with torch.no_grad():
            outputs = trained_model(inputs)
            score = outputs.data
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend((score > 0.5).cpu().numpy().tolist())


test_acc = accuracy_score(y_true, y_pred)

print(f"Test set accuracy is: {test_acc:4f}")
