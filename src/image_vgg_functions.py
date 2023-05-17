import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.applications import resnet_v2
from keras.applications import efficientnet


# Function that imports the images with/without augmenting the data
# Returns the training and validation (if valid=True) sets
# Preprocesses the data according to preprocess_input function for the architecture that is used
def load_images(path, batch_size, generate_data=False, valid=True, architecture=None):
    # The names of the classes according to the names of the directories
    classes = sorted(os.listdir(path))[1:]

    preprocess = None
    rescaling = None
    sample_mean = False
    sample_std = False

    # Check if preprocessing is for a simple CNN or for a specific architecture

    if architecture == "toy_cnn":
        rescaling = 1.0 / 255
        img_size = 224
    elif architecture == "vgg":
        preprocess = vgg16.preprocess_input
        img_size = 224
    elif architecture == "resnet":
        preprocess = resnet_v2.preprocess_input
        img_size = 224
    elif architecture == "efficientnet":
        preprocess = efficientnet.preprocess_input
        img_size = 380
    elif architecture == "vit":
        rescaling = 1.0 / 255
        sample_mean = True
        sample_std = True
        img_size = 250
    elif architecture == "pretrained_vit":
        rescaling = 1.0 / 255
        sample_mean = True
        sample_std = True
        img_size = 224
    else:
        os.exit("Unknown architecture input in load_images")

    if valid:
        datagen = ImageDataGenerator(
            horizontal_flip=generate_data,
            vertical_flip=generate_data,
            validation_split=0.2,
            preprocessing_function=preprocess,
            rescale=rescaling,
            samplewise_center=sample_mean,
            samplewise_std_normalization=sample_std,
        )

        train_set = datagen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            subset="training",
            class_mode="categorical",
            classes=classes,
            shuffle=True,
        )

        valid_set = datagen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            subset="validation",
            class_mode="categorical",
            classes=classes,
            shuffle=True,
        )
    else:
        datagen = ImageDataGenerator(
            horizontal_flip=generate_data,
            vertical_flip=generate_data,
            preprocessing_function=preprocess,
            rescale=rescaling,
            samplewise_center=sample_mean,
            samplewise_std_normalization=sample_std,
        )

        train_set = datagen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="categorical",
            classes=classes,
            shuffle=True,
        )

        valid_set = None

    return train_set, valid_set


# Loads the preprocessed xtrain set and the labels ytrain
def load_x_and_y_train(path, architecture=None):
    batch_size = 1000
    train_set, _ = load_images(path, batch_size, valid=False, architecture=architecture)
    sample_count = train_set.samples

    if architecture == "efficientnet":
        img_size = 380
    elif architecture == "vit":
        img_size = 250
    else:
        img_size = 224

    x_train = np.zeros(shape=(sample_count, img_size, img_size, 3))
    y_train = np.zeros(shape=(sample_count, 4))

    i = 0
    for x, y in train_set:
        x_train[i * batch_size : (i + 1) * batch_size] = x
        y_train[i * batch_size : (i + 1) * batch_size] = y
        i += 1
        if i * batch_size >= sample_count:
            break

    return x_train, y_train


# Find the candidate unlabelled images
def find_candidate_images(images_path):
    """
    Finds all candidate images in the given folder and its sub-folders.

    Returns:
        images: a list of absolute paths to the discovered images.
    """
    images = []
    for root, _, files in os.walk(images_path):
        for name in files:
            file_path = os.path.abspath(os.path.join(root, name))
            if (os.path.splitext(name)[1]).lower() in [".jpg", ".png", ".jpeg"]:
                images.append(file_path)
    return images


# Returns the preprocessed numpy array of an image
def get_image(path, input_shape, architecture=None):
    img = keras.utils.load_img(path, target_size=input_shape)
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if architecture == "vgg":
        x = vgg16.preprocess_input(x)
    elif architecture == "resnet":
        x = resnet_v2.preprocess_input(x)
    elif architecture == "efficientnet":
        x = efficientnet.preprocess_input(x)
    elif architecture == "toy_cnn":
        x *= 1.0 / 255
    else:
        os.exit("Unknown architecture input in get_image")

    return x
