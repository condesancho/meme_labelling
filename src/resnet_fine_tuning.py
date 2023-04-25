from keras.models import Model
from keras.applications.resnet_v2 import ResNet152V2
from keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

import os
import time

from image_vgg_functions import load_images, load_x_and_y_train

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PATH = "../data/categories/"
BATCH_SIZE = 64


def mlp(x, hidden_units, dropout_rate):
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def model_builder(hp):
    hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    hp_drop = hp.Choice("dropout_rate", values=[0.5, 0.6, 0.75])
    hp_trainable = hp.Choice("resnet_trainable_layers", values=[10, 15, 20])

    resnet = ResNet152V2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg"
    )

    # Leave last layers for fine tuning
    for layer in resnet.layers[:-hp_trainable]:
        layer.trainable = False
    # Of the last layers that we unfroze we leave BatchNorm layers frozen
    for layer in resnet.layers[-hp_trainable:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Construct the top model and the output
    top_model = mlp(resnet.output, [1024, 512], hp_drop)
    output_layer = layers.Dense(4, activation="softmax")(top_model)

    # Final model
    model = Model(inputs=resnet.input, outputs=output_layer)

    optim = keras.optimizers.Adam(learning_rate=hp_lr)

    # Compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]
    )

    return model


x_train, y_train = load_x_and_y_train(PATH, architecture="resnet")

# Initialize tuner
tuner = kt.Hyperband(
    model_builder, objective="val_accuracy", max_epochs=20, directory="resnet_tuning"
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
print(
    "No. optimal trainable ResNet152V2 layers:", best_hps.get("resnet_trainable_layers")
)
print("Dropout rate:", best_hps.get("dropout_rate"))
print("Learning rate:", best_hps.get("learning_rate"))


### Train the optimal model ###
resnet = ResNet152V2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# Leave last layers for fine tuning
fine_tune = best_hps.get("resnet_trainable_layers")  # no. layers to fine tune
for layer in resnet.layers[:-fine_tune]:
    layer.trainable = False
for layer in resnet.layers[-fine_tune:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Construct the top model and the output
top_model = mlp(resnet.output, [1024, 512], best_hps.get("dropout_rate"))
output_layer = layers.Dense(4, activation="softmax")(top_model)

# Final model
model = Model(inputs=resnet.input, outputs=output_layer)

print(model.summary())

# Load the images
train, valid = load_images(
    PATH,
    BATCH_SIZE,
    generate_data=True,
    valid=True,
    architecture="resnet",
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

resnet_history = model.fit(
    train,
    batch_size=BATCH_SIZE,
    epochs=n_epochs,
    validation_data=valid,
    steps_per_epoch=n_steps,
    validation_steps=n_val_steps,
    callbacks=[early_stop],
    verbose=1,
)

resnet_output = layers.GlobalAveragePooling2D(name="avg_pool")(resnet.output)
resnet_with_avg_pool = Model(inputs=resnet.input, outputs=resnet_output)
resnet_with_avg_pool.save("../models/resnet_feat_ex/resnet_fine_tuned.h5")

model.save("../models/resnet/model.h5")
