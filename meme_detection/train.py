import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score

from models import CustomCNN, CustomResNet
from data import ImageDataset

import time
import random
import numpy as np
import pickle
import json

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    raise Exception("No GPUs available!")

data_dir = "../data"
imnet_mean = [0.485, 0.456, 0.406]
imnet_std = [0.229, 0.224, 0.225]
IMG_SIZE = 224
batch_size = 128
epochs = 10

results = []
train_ds = ImageDataset(
    directory=data_dir,
    split="train",
    reg_img_split="coco",
    transform=transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imnet_mean, imnet_std),
        ]
    ),
)
train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)
val_ds = ImageDataset(
    directory=data_dir,
    split="val",
    reg_img_split="coco",
    transform=transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(imnet_mean, imnet_std),
        ]
    ),
)
val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)

learning_rate = 1e-4
dropout = 0.5
depth = 5

ckpt = f"ckpt/cnn.pt"
max_auc = 0

model = CustomCNN(depth, dropout)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start = time.time()
val_loss = []
val_acc = []
val_auc = []
for epoch in range(epochs):
    print(f"Epoch {epoch}/{epochs - 1}")
    print("-" * 10)
    model.train()
    for data, labels in train_dl:
        labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss_ = criterion(outputs, labels.float().view(-1, 1))
        loss_.backward()
        optimizer.step()

    model.eval()
    val_loss_ = 0
    y_score = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, labels in val_dl:
            labels.to(device)
            outputs = model(data)
            val_loss_ += criterion(outputs, labels.float().view(-1, 1)).item()
            score = outputs.data
            y_score.extend(score.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend((score > 0.5).cpu().numpy().tolist())

    val_loss_ /= len(val_dl)
    val_acc_ = accuracy_score(y_true, y_pred)
    val_auc_ = roc_auc_score(y_true, y_score)

    print(f"Validation Loss: {val_loss_:.4f} Acc: {val_acc_:.4f}")

    # if val_auc_ > max_auc:
    #     max_auc = val_auc_
    #     torch.save(model, ckpt)
    #     with open(f"{ckpt[:-3]}.txt", "w") as file:
    #         file.write(
    #             json.dumps(
    #                 {
    #                     "modality": modality,
    #                     "lr": learning_rate,
    #                     "pretrained": pretrained_,
    #                     "hidden_dim": hidden_dim,
    #                     "lstm_layers": lstm_layers_,
    #                     "val_loss": val_loss_,
    #                     "val_acc": val_acc_,
    #                     "val_auc": val_auc_,
    #                     "epoch": epoch,
    #                 }
    #             )
    #         )

    val_loss.append(val_loss_)
    val_acc.append(val_acc_)
    val_auc.append(val_auc_)

    if epoch == int(epochs / 2) - 1:
        for g in optimizer.param_groups:
            g["lr"] = learning_rate / 10
