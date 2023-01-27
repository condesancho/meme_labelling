from keras import optimizers
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os

from image_vgg_functions import load_images

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PATH = "./data/categories/"
BATCH_SIZE = 64


def mlp(x, hidden_units, dropout_rate):
    x = layers.Flatten(name="flatten")(x)
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


train, valid = load_images(PATH, BATCH_SIZE, generate_data=True, valid=True)

vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Leave last layers for fine tuning
fine_tune = 4  # no. layers to fine tune
for layer in vgg.layers[:-fine_tune]:
    layer.trainable = False

# Construct the top model and the output
top_model = mlp(vgg.output, [4096, 1024], 0.2)
output_layer = layers.Dense(4, activation="softmax")(top_model)

# Final model
model = Model(inputs=vgg.input, outputs=output_layer)

# Model parameters
n_epochs = 1
lr = 0.0001  # Learning rate
optim = keras.optimizers.Adam(learning_rate=lr)
n_steps = train.samples // BATCH_SIZE

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

tl_checkpoint = ModelCheckpoint(
    filepath="tl_model_v1.weights.best.hdf5", save_best_only=True, verbose=1
)

if valid is not None:
    n_val_steps = valid.samples // BATCH_SIZE

    # EarlyStopping
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
    )

    vgg_history = model.fit(
        train,
        batch_size=BATCH_SIZE,
        epochs=n_epochs,
        validation_data=valid,
        steps_per_epoch=n_steps,
        validation_steps=n_val_steps,
        callbacks=[tl_checkpoint, early_stop],
        verbose=1,
    )
else:
    vgg_history = model.fit(
        train,
        batch_size=BATCH_SIZE,
        epochs=n_epochs,
        steps_per_epoch=n_steps,
        callbacks=[tl_checkpoint],
        verbose=1,
    )

vgg.save("vgg_fine_tuned.h5")
