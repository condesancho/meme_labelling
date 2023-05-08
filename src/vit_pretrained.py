import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

from datasets import load_dataset
from transformers import ViTImageProcessor

from vit_torch_classes import CustomVitModel, EarlyStopping
import time
import copy

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "google/vit-base-patch16-224-in21k"
DATA_DIR = "../data/torch"
BATCH_SIZE = 64

""" Loading the image data """
train_ds = load_dataset("imagefolder", data_dir=DATA_DIR, split="train")

# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.2)
train_ds = splits["train"]
val_ds = splits["test"]


""" Preprocessing the data """
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# Data augmentations
normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
    [
        Resize((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        normalize,
    ]
)

_val_transforms = Compose(
    [
        Resize((size, size)),
        ToTensor(),
        normalize,
    ]
)


def train_transforms(examples):
    examples["pixel_values"] = [
        _train_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


def val_transforms(examples):
    examples["pixel_values"] = [
        _val_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)

image_datasets = {"train": train_ds, "val": val_ds}


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}


""" Define the training function """


def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    valid_acc = 0.0

    # early stopping
    early_stopping = EarlyStopping()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for item in dataloaders[phase]:
                inputs = item["pixel_values"].to(device)
                labels = item["labels"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                valid_acc = epoch_acc

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        early_stopping(valid_acc)
        if early_stopping.early_stop:
            print("Early stopping... We are at epoch:", epoch)
            break

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


# Define hyperparameters
hp_lr = [1e-2, 1e-3, 1e-4]
hp_dropout = [0.5, 0.6, 0.75]
hp_weight_decay = [1e-2, 1e-3, 1e-4]

best_lr = 0.0
best_dropout = 0.0
best_weight_decay = 0.0
best_acc = 0.0
for lr in hp_lr:
    for dropout in hp_dropout:
        for weight_decay in hp_weight_decay:
            print(
                f"Running test for {lr} learning rate, {dropout} dropout and {weight_decay} weight_decay"
            )
            model = CustomVitModel(dropout=dropout)
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            _, model_acc = train_model(model, criterion, optimizer, exp_lr_scheduler)

            if model_acc > best_acc:
                best_acc = model_acc
                best_lr = lr
                best_dropout = dropout
                best_weight_decay = weight_decay

            del model
            torch.cuda.empty_cache()

print("The best hyperparameters are:")
print(f"Learning rate: {best_lr}")
print(f"Dropout probability: {best_dropout}")
print(f"Weight decay: {best_weight_decay}")


""" Train the model for the best hyperparameters """
# model = CustomVitModel()
# model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# model = train_model(model, criterion, optimizer, exp_lr_scheduler)

# PATH = "../models/vit_pretrained/model.pth"

# torch.save(model.state_dict(), PATH)
