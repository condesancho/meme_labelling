import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
import keras_tuner as kt

import os
import time

import numpy as np

from image_vgg_functions import load_images, load_x_and_y_train

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PATH = "../data/categories/"
BATCH_SIZE = 64


def mlp(x, hidden_units, dropout_rate):
    x = layers.Flatten(name="flatten")(x)
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def cnn(filters=[(64, 2), (128, 1), (128, 1)], pool_size=2):
    conv_model = models.Sequential()
    conv_model.add(
        layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(224, 224, 3), padding="same"
        )
    )
    conv_model.add(layers.BatchNormalization())
    conv_model.add(layers.MaxPooling2D((pool_size, pool_size)))
    for n_filters, n_convs in filters:
        for _ in np.arange(n_convs):
            conv_model.add(
                layers.Conv2D(n_filters, (3, 3), activation="relu", padding="same")
            )
        conv_model.add(layers.BatchNormalization())
        conv_model.add(layers.MaxPooling2D((pool_size, pool_size)))

    return conv_model


def model_builder(hp):
    hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    hp_drop = hp.Choice("dropout_rate", values=[0.5, 0.6, 0.75])
    hp_depth = hp.Choice("conv_depth", values=[4, 5, 6])
    hp_hidden_1 = hp.Choice("hidden_1", values=[1024, 4096])
    hp_hidden_2 = hp.Choice("hidden_2", values=[1024, 512])

    if hp_depth == 4:
        filters = [(64, 2), (128, 1), (128, 1)]
    elif hp_depth == 5:
        filters = [(64, 1), (128, 2), (256, 1), (256, 1)]
    elif hp_depth == 6:
        filters = [(64, 1), (128, 2), (256, 2), (256, 1)]

    conv_model = cnn(filters=filters, pool_size=2)

    # Construct the top model and the output
    top_model = mlp(conv_model.output, [hp_hidden_1, hp_hidden_2], hp_drop)
    output_layer = layers.Dense(4, activation="softmax")(top_model)

    # Final model
    model = Model(inputs=conv_model.input, outputs=output_layer)

    optim = keras.optimizers.Adam(learning_rate=hp_lr)

    # Compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]
    )

    return model


x_train, y_train = load_x_and_y_train(PATH, architecture='toy_cnn')

# Initialize tuner
tuner = kt.Hyperband(
    model_builder, objective="val_accuracy", max_epochs=20, directory="cnn_tuning"
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
print("Convolutional depth:", best_hps.get("conv_depth"))
print("No. neurons in first hidden layer:", best_hps.get("hidden_1"))
print("No. neurons in second hidden layer:", best_hps.get("hidden_2"))

# Get the best cnn model based on the depth above
if best_hps.get("conv_depth") == 4:
    filters = [(64, 2), (128, 1), (128, 1)]
elif best_hps.get("conv_depth") == 5:
    filters = [(64, 1), (128, 2), (256, 1), (256, 1)]
elif best_hps.get("conv_depth") == 6:
    filters = [(64, 1), (128, 2), (256, 2), (256, 1)]

### Train the optimal model ###
conv_model = cnn(filters=filters)

# Construct the top model and the output
top_model = mlp(
    conv_model.output,
    [best_hps.get("hidden_1"), best_hps.get("hidden_2")],
    best_hps.get("dropout_rate"),
)
output_layer = layers.Dense(4, activation="softmax")(top_model)

# Final model
model = Model(inputs=conv_model.input, outputs=output_layer)

print(model.summary())

# Load the images with preprocessing for the cnn
train, valid = load_images(
    PATH, BATCH_SIZE, generate_data=True, valid=True, architecture='toy_cnn'
)

# Model parameters
n_epochs = 20
lr = best_hps.get("learning_rate")  # Learning rate
optim = keras.optimizers.Adam(learning_rate=lr)
n_steps = train.samples // BATCH_SIZE

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

n_val_steps = valid.samples // BATCH_SIZE

# EarlyStopping
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
)

cnn_history = model.fit(
    train,
    batch_size=BATCH_SIZE,
    epochs=n_epochs,
    validation_data=valid,
    steps_per_epoch=n_steps,
    validation_steps=n_val_steps,
    callbacks=[early_stop],
    verbose=1,
)

conv_model.save("../models/cnn_feat_ex/cnn_fine_tuned.h5")

model.save("../models/cnn/model.h5")
