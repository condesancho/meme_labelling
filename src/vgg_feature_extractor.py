from tensorflow.keras.models import load_model

import os
from os.path import join
import numpy as np
import pandas as pd

from image_vgg_functions import load_images, find_candidate_images, get_image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

UNLABELLED_PATH = "../data/images/"
LABELLED_PATH = "../data/categories/"
MODEL_PATH = "../models/vgg_feat_ex/"
BATCH_SIZE = 64

# Import the labelled data
labelled_data, _ = load_images(
    LABELLED_PATH, BATCH_SIZE, generate_data=False, valid=False
)

# Import the fine tuned model
ft_model = load_model(MODEL_PATH + "vgg_fine_tuned.h5")

### Feature extraction for the labelled images ###
sample_count = labelled_data.samples

labelled_features = np.zeros(shape=(sample_count, 7, 7, 512))
labels = np.zeros(shape=(sample_count, 4))

i = 0
for inputs_batch, labels_batch in labelled_data:
    features_batch = ft_model.predict(inputs_batch)
    labelled_features[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = features_batch
    labels[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = labels_batch
    i += 1
    if i * BATCH_SIZE >= sample_count:
        break


df_labelled = pd.DataFrame(data=np.reshape(labelled_features, (sample_count, -1)))
labels = np.nonzero(labels)[1] + 1
df_labelled["label"] = labels
df_labelled.to_csv(MODEL_PATH + "labelled_features.csv", index=False)


### Feature extraction for the unlabelled images ###
input_shape = ft_model.input_shape[1:3]
# get images
candidate_images = find_candidate_images(UNLABELLED_PATH)
# analyze images and grab activations
activations = []
unlabelled_imgs = []
for idx, image_path in enumerate(candidate_images):
    file_path = join(UNLABELLED_PATH, image_path)
    img = get_image(file_path, input_shape, vgg=True)
    if img is not None:
        if idx % 100 == 0:
            print(
                "getting activations for %s %d/%d"
                % (image_path, idx, len(candidate_images))
            )
        acts = ft_model.predict(img)[0]

        activations.append(np.reshape(acts, -1))
        unlabelled_imgs.append(image_path)


unlabelled_features = np.array(activations)

df_unlabelled = pd.DataFrame(data=unlabelled_features)
df_unlabelled["image"] = unlabelled_imgs
df_unlabelled.to_csv(MODEL_PATH + "unlabelled_features.csv", index=False)
