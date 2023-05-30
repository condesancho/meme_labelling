import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from sklearn.metrics import accuracy_score

import time
import copy

import pickle

from utils import EarlyStopping, vit_experiments, resnet_experiments

from models import CustomResNet, CustomVitModel


""" Define the training function """


def train_model(
    model,
    image_datasets,
    batch_size,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=20,
):
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    valid_acc = 0.0

    val_acc_history = []
    val_loss_history = []

    # early stopping
    early_stopping = EarlyStopping(monitor="accuracy")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            y_true = []
            y_pred = []

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
                    loss = criterion(outputs, labels.float().view(-1, 1))
                    score = outputs.data
                    y_true.extend(labels.cpu().numpy().tolist())
                    y_pred.extend((score > 0.5).cpu().numpy().tolist())

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy_score(y_true, y_pred)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                valid_loss = epoch_loss
                valid_acc = epoch_acc
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

            # deep copy the model
            if phase == "val" and (epoch_loss < min(val_loss_history) or epoch == 0):
                best_model_wts = copy.deepcopy(model.state_dict())

        early_stopping(valid_acc)
        last_epoch = epoch + 1
        if early_stopping.early_stop:
            print(f"Early stopping... We are at epoch:{last_epoch}")
            break

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {max(val_acc_history):4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, last_epoch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


""" Define the function for the hyperparmeter tuning """


def hyperparam_tuning(model_name, image_datasets, batch_size, device, save_filepath):
    results = []
    epochs = 10

    if model_name == "resnet":
        experiments = resnet_experiments()
        for lr, dropout, trainable_layers in experiments:
            print("Running experiment for:")
            print(
                f"{lr} learning rate, {dropout} dropout and {trainable_layers} trainable layers"
            )
            model = CustomResNet(trainable_layers, dropout)

            model.to(device)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

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
                epochs,
            )

            # Deallocate memory
            del model
            del optimizer
            torch.cuda.empty_cache()

            config = {
                "batch_size": batch_size,
                "lr": lr,
                "dropout": dropout,
                "trainable_layers": trainable_layers,
                "epochs": epochs,
                "epochs_used_for_training": last_epoch,
            }

            max_acc = max(val_acc_history)
            print(
                f"For {lr} learning rate, {dropout} dropout and {trainable_layers} trainable layers"
            )
            print(f"the max val accuracy was: {max_acc}\n")

            results.append({"config": config, "accuracy": max_acc})
            with open(save_filepath, "wb") as h:
                pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    elif model_name == "vit":
        experiments = vit_experiments()
        for lr, dropout, weight_decay in experiments:
            print("Running experiment for:")
            print(
                f"{lr} learning rate, {dropout} dropout and {weight_decay} weight decay"
            )
            model = CustomVitModel(dropout)
            model.to(device)

            criterion = nn.BCELoss()
            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
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
                epochs,
            )
            del model
            del optimizer
            torch.cuda.empty_cache()

            config = {
                "batch_size": batch_size,
                "lr": lr,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "epochs_used_for_training": last_epoch,
            }

            max_acc = max(val_acc_history)
            print(
                f"For {lr} learning rate, {dropout} dropout and {weight_decay} weight decay"
            )
            print(f"the max val accuracy was: {max_acc}\n")

            results.append({"config": config, "accuracy": max_acc})
            with open(save_filepath, "wb") as h:
                pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        error = ValueError(
            "hyperparam_tuning: model_name does not match an appropriate model name"
        )
        raise error
