import os
import numpy as np

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input


# Function that imports the images with/without augmenting the data
# Returns the training and validation (if valid=True) sets
def load_images(path, batch_size, generate_data=False, valid=True, vgg=True):
    # The names of the classes according to the names of the directories
    classes = sorted(os.listdir(path))[1:]

    if valid:
        # Check if preprocessing is for the VGG or the CNN
        if vgg:
            # This imports only the images from the above classes
            train_datagen = ImageDataGenerator(
                horizontal_flip=generate_data,
                vertical_flip=generate_data,
                validation_split=0.2,
                preprocessing_function=preprocess_input,
            )  # VGG16 preprocessing
        else:
            train_datagen = ImageDataGenerator(
                horizontal_flip=generate_data,
                vertical_flip=generate_data,
                validation_split=0.2,
                rescale=1.0 / 255,
            )  # regular preprocessing

        train_set = train_datagen.flow_from_directory(
            path,
            target_size=(224, 224),
            batch_size=batch_size,
            subset="training",
            class_mode="categorical",
            classes=classes,
            shuffle=True,
        )

        valid_set = train_datagen.flow_from_directory(
            path,
            target_size=(224, 224),
            batch_size=batch_size,
            subset="validation",
            class_mode="categorical",
            classes=classes,
            shuffle=True,
        )
    else:
        if vgg:
            train_datagen = ImageDataGenerator(
                horizontal_flip=generate_data,
                vertical_flip=generate_data,
                preprocessing_function=preprocess_input,
            )  # VGG16 preprocessing
        else:
            train_datagen = ImageDataGenerator(
                horizontal_flip=generate_data,
                vertical_flip=generate_data,
                rescale=1.0 / 255,
            )  # regular preprocessing

        train_set = train_datagen.flow_from_directory(
            path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            classes=classes,
            shuffle=True,
        )

        valid_set = None

    return train_set, valid_set


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


def get_image(path, input_shape):
    img = keras.utils.load_img(path, target_size=input_shape)
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
