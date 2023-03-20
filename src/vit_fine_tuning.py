from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import tensorflow_addons as tfa

import os
import time

from image_vgg_functions import load_images, load_x_and_y_train
from vit import PatchEncoder, Patches, PositionalEmbedding, vit

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PATH = "../data/categories/"
BATCH_SIZE = 64

img_size = 250
input_shape = (img_size, img_size, 3)
patch_size = 25
num_patches = (img_size // patch_size) ** 2

projection_dim = 64
transformer_layers = 8
transformer_units = [projection_dim * 2, projection_dim]
num_heads = 4
mlp_head_units = [2048, 1024]


def model_builder(hp):
    hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    hp_drop = hp.Choice("dropout_rate", values=[0.5, 0.6, 0.75])
    hp_weight_decay = hp.Choice("weight_decay", values=[1e-2, 1e-3, 1e-4])

    model, _ = vit(
        input_shape,
        patch_size,
        projection_dim,
        num_patches,
        transformer_layers,
        num_heads,
        transformer_units,
        mlp_head_units,
        hp_drop,
    )

    optim = tfa.optimizers.AdamW(learning_rate=hp_lr, weight_decay=hp_weight_decay)

    # Compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]
    )

    return model


x_train, y_train = load_x_and_y_train(PATH, architecture="vit")

# Initialize tuner
tuner = kt.Hyperband(
    model_builder, objective="val_accuracy", max_epochs=20, directory="vit_tuning"
)

early_stop = EarlyStopping(monitor="val_loss", patience=5)

st = time.time()

tuner.search(
    x_train,
    y_train,
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stop],
)

end = time.time()

print("Hyperparametre search time:", end - st, "seconds")

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("The chosen optimal parameters are:")
print("Dropout rate:", best_hps.get("dropout_rate"))
print("Learning rate:", best_hps.get("learning_rate"))
print("Weight decay:", best_hps.get("weight_decay"))

### Train the optimal model ###
# Final model
model, vit_model = vit(
    input_shape,
    patch_size,
    projection_dim,
    num_patches,
    transformer_layers,
    num_heads,
    transformer_units,
    mlp_head_units,
    best_hps.get("dropout_rate"),
)

# Load the images
train, valid = load_images(
    PATH, BATCH_SIZE, generate_data=True, valid=True, architecture="vit"
)

# Model parameters
n_epochs = 20
lr = best_hps.get("learning_rate")  # Learning rate
wd = best_hps.get("weight_decay")  # Weight decay
optim = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
n_steps = train.samples // BATCH_SIZE

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

n_val_steps = valid.samples // BATCH_SIZE

# EarlyStopping
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
)

vit_history = model.fit(
    train,
    batch_size=BATCH_SIZE,
    epochs=n_epochs,
    validation_data=valid,
    steps_per_epoch=n_steps,
    validation_steps=n_val_steps,
    callbacks=[early_stop],
    verbose=1,
)

vit_model.save("../models/vit_feat_ex/vit_fine_tuned.h5")

model.save("../models/vit/model.h5")
