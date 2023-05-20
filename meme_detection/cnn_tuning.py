import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score

from data import ImageDataset
from models import CustomCNN, CustomResNet

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

# Hyperparameters
hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
hp_drop = hp.Choice("dropout_rate", values=[0.5, 0.6, 0.75])
hp_depth = hp.Choice("conv_depth", values=[4, 5, 6])
hp_hidden_1 = hp.Choice("hidden_1", values=[1024, 4096])
hp_hidden_2 = hp.Choice("hidden_2", values=[1024, 512])
