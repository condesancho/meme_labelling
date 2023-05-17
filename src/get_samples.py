import os
import shutil
import random
import pandas as pd
import math

""" Code that takes some samples from the whole dataset and copies them to another folder """
# img_dir = "../data/images/"

# dest_dir = "../data/img_samples/"

# # Takes random samples
# filenames = random.sample(os.listdir(img_dir), 500)

# for file in filenames:
#     src_path = os.path.join(img_dir, file)
#     dst_path = os.path.join(dest_dir, file)
#     shutil.copyfile(src_path, dst_path)


""" Code that moves the images that TextFuseNet found no text to another folder """
# dest_dir = "../data/no_text/"

# df = pd.read_csv("../text_detection/textfusenet_txt_presence.csv", index_col=0)

# no_text = df[df.loc[:, "text"] == False]

# for idx, row in no_text.iterrows():
#     shutil.move(row["image"], dest_dir)

""" Code that moves images from the 4 labelled categories to split a training and validation set """
data_dir = "../data/torch/"

count = 0
for dir in sorted(os.listdir(data_dir))[:4]:
    # Create train directory with its subdirectories
    train_dir = data_dir + "train/" + dir
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Create validation dir
    valid_dir = data_dir + "valid/"
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    img_dir = os.path.join(data_dir, dir)

    # Count the number of files in the initial directory
    for img in os.listdir(img_dir):
        if os.path.isfile(os.path.join(img_dir, img)):
            count += 1
    train_size = math.floor(0.8 * count)

    # Takes random samples and moves them to the train dir
    filenames = random.sample(os.listdir(img_dir), train_size)
    for file in filenames:
        src_path = os.path.join(data_dir, dir, file)
        dst_path = os.path.join(train_dir, file)
        shutil.move(src_path, dst_path)

    # Move the remaining images to the validation directory
    shutil.move(os.path.join(data_dir, dir), valid_dir)

    # Reset count
    count = 0
