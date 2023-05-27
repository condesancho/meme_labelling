import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

import numpy as np

from transformers import ViTImageProcessor

import argparse
import sys
import os
import pickle
import random
import warnings

warnings.filterwarnings("ignore")

from train import train_model, hyperparam_tuning
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
    parser.add_argument(
        "--hp_tuning",
        action="store_true",
        help="whether or not to apply hyperparameter tuning to the model among some hyperparameters",
    )
    parser.add_argument("--no-hp_tuning", dest="hp_tuning", action="store_false")
    parser.set_defaults(hp_tuning=False)
    params = vars(parser.parse_args(args))
    return params


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    raise Exception("No GPUs available!")

params = process_arguments(sys.argv[1:])
dataset_split = params["dataset_split"]
model_name = params["model"]
hp_tuning = params["hp_tuning"]


data_dir = "../data"  # directory where the folder meme_detection_dataset is stored

save_path = os.path.join("./results", model_name, dataset_split)
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_file = os.path.join(save_path, "hyperparam_tuning.pickle")

# Training parameters
imnet_mean = [0.485, 0.456, 0.406]
imnet_std = [0.229, 0.224, 0.225]
if model_name == "resnet":
    IMG_SIZE = 224
else:
    vit_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(vit_name)
    IMG_SIZE = processor.size["height"]
batch_size = 128

# Create train and validation datasets
train_ds = ImageDataset(
    directory=data_dir,
    split="train",
    reg_img_split=dataset_split,
    transform=Compose(
        [
            Resize((IMG_SIZE, IMG_SIZE)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Normalize(imnet_mean, imnet_std),
        ]
    ),
)
val_ds = ImageDataset(
    directory=data_dir,
    split="val",
    reg_img_split=dataset_split,
    transform=Compose(
        [
            Resize((IMG_SIZE, IMG_SIZE)),
            ToTensor(),
            Normalize(imnet_mean, imnet_std),
        ]
    ),
)

image_datasets = {"train": train_ds, "val": val_ds}

# Hyperparameter tuning
if hp_tuning:
    hyperparam_tuning(
        model_name=model_name,
        image_datasets=image_datasets,
        batch_size=batch_size,
        device=device,
        save_filepath=save_file,
    )
else:
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
    if model_name == "resnet":
        print(f"Trainable layers:{best_config['trainable_layers']}")

        model = CustomResNet(best_config["trainable_layers"], best_config["dropout"])
        model.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_config["lr"])

        # Decay LR by a factor of 0.1 every 5 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        model, val_acc_history, last_epoch = train_model(
            model,
            image_datasets,
            batch_size,
            criterion,
            optimizer,
            exp_lr_scheduler,
            device,
        )
    else:
        print(f"Weight decay:{best_config['weight_decay']}")
        model = CustomVitModel(best_config["dropout"])
        model.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
        )
        # Decay LR by a factor of 0.1 every 5 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        model, val_acc_history, last_epoch = train_model(
            model,
            image_datasets,
            batch_size,
            criterion,
            optimizer,
            exp_lr_scheduler,
            device,
        )
    best_model_file = os.path.join(save_path, "best_model.pth")
    torch.save(model.state_dict(), best_model_file)

    acc = {"val_acc": val_acc_history}
    results_path = os.path.join(save_path, "results.pickle")
    with open(results_path, "wb") as h:
        pickle.dump(acc, h, protocol=pickle.HIGHEST_PROTOCOL)
