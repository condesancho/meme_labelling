import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow import keras
from keras.models import Model
from keras.callbacks import EarlyStopping

from image_vgg_functions import load_images

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PATH = "../data/categories/"
BATCH_SIZE = 64


def mlp(x, hidden_units, dropout_rate):
    x = layers.Flatten(name="flatten")(x)
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def cnn(filters_convs=[(64, 2), (128, 1), (128, 1)], pool_size=2):
    conv_model = models.Sequential()
    conv_model.add(
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(224, 224, 3), padding="same"
        )
    )
    conv_model.add(layers.MaxPooling2D((pool_size, pool_size)))
    for n_filters, n_convs in filters_convs:
        for _ in np.arange(n_convs):
            conv_model.add(
                layers.Conv2D(n_filters, (3, 3), activation="relu", padding="same")
            )
        conv_model.add(layers.MaxPooling2D((pool_size, pool_size)))

    return conv_model


# Load the images with preprocessing for the cnn
train, valid = load_images(PATH, BATCH_SIZE, generate_data=True, valid=True, vgg=False)


conv_model = cnn()

# Construct the top model and the output
top_model = mlp(conv_model.output, [4096, 1024], 0.2)
output_layer = layers.Dense(4, activation="softmax")(top_model)

# Final model
model = Model(inputs=conv_model.input, outputs=output_layer)

print(model.summary())

# Model parameters
n_epochs = 20
lr = 0.0001  # Learning rate
optim = keras.optimizers.Adam(learning_rate=lr)
n_steps = train.samples // BATCH_SIZE

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

if valid is not None:
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
else:
    cnn_history = model.fit(
        train,
        batch_size=BATCH_SIZE,
        epochs=n_epochs,
        steps_per_epoch=n_steps,
        verbose=1,
    )

conv_model.save("./cnn/cnn_fine_tuned.h5")
