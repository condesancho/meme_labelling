from tensorflow.keras.models import load_model

import os
from os.path import join
import numpy as np
import pandas as pd
import sys

from image_vgg_functions import find_candidate_images, get_image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

UNLABELLED_PATH = "../data/images/"

selection = int(
    input("Select 1 for CNN, 2 for VGG, 3 for ResNet or 4 for EfficientNet: ")
)
if selection == 1:
    model_dir = "../models/cnn/"
    arch = "toy_cnn"
elif selection == 2:
    model_dir = "../models/vgg/"
    arch = "vgg"
elif selection == 3:
    model_dir = "../models/resnet/"
    arch = "resnet"
elif selection == 4:
    model_dir = "../models/efficientnet/"
    arch = "efficientnet"
else:
    sys.exit("Invalid input. Try again")

trained_model = load_model(model_dir + "model.h5")

### Feature extraction for the unlabelled images ###
input_shape = trained_model.input_shape[1:3]
# get images
candidate_images = find_candidate_images(UNLABELLED_PATH)
# analyze images and grab predictions
predictions = []
unlabelled_imgs = []
for idx, image_path in enumerate(candidate_images):
    file_path = join(UNLABELLED_PATH, image_path)
    img = get_image(file_path, input_shape, architecture=arch)
    if img is not None:
        if idx % 100 == 0:
            print(
                "getting prediction for %s %d/%d"
                % (image_path, idx, len(candidate_images))
            )
        y_prob = trained_model.predict(img)

        predictions.append(y_prob.argmax() + 1)
        unlabelled_imgs.append(image_path)

df = pd.DataFrame()
df["image"] = unlabelled_imgs
df["predicted_labels"] = predictions
df.to_csv(model_dir + "predicted_labels.csv", index=False)
