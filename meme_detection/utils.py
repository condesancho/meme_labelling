import os
import shutil
import numpy as np
import pandas as pd
import random
import sys


def split_dataset(data_dir, reg_img_split: str):
    meme_dir = os.path.join(data_dir, "meme_detection_dataset/memes")
    reg_img_dir = os.path.join(data_dir, "meme_detection_dataset/regular_images")

    coco_path = os.path.join(reg_img_dir, "coco")
    icdar_path = os.path.join(reg_img_dir, "icdar")
    cc_path = os.path.join(reg_img_dir, "conceptual_captions")

    df_columns = ["file_path", "label", "subdir"]

    files_memes = pd.DataFrame(get_files(meme_dir, label=0), columns=df_columns)

    files_coco = pd.DataFrame(get_files(coco_path, label=1), columns=df_columns)
    files_icdar = pd.DataFrame(get_files(icdar_path, label=1), columns=df_columns)
    files_cc = pd.DataFrame(get_files(cc_path, label=1), columns=df_columns)

    n = len(files_memes)

    if reg_img_split == "coco":
        n_coco = int(0.5 * n)
        n_cc = int(0.25 * n)
        n_icdar = n - (n_coco + n_cc)
    elif reg_img_split == "conc_capt":
        n_coco = int(0.25 * n)
        n_cc = int(0.5 * n)
        n_icdar = n - (n_coco + n_cc)
    elif reg_img_split == "icdar":
        n_coco = int(0.25 * n)
        n_cc = int(0.25 * n)
        n_icdar = n - (n_coco + n_cc)
    elif reg_img_split == "equally":
        n_coco = int((1 / 3) * n)
        n_cc = int((1 / 3) * n)
        n_icdar = n - (n_coco + n_cc)
    else:
        sys.exit("Regular Image Split not valid")

    # Select 50% - 50% text presence and text absence from COCO dataset
    n_coco_no_txt = int(0.5 * n_coco)
    n_coco_txt = n_coco - n_coco_no_txt
    coco_txt_sample = files_coco.loc[files_coco["subdir"] == "text_presence"].sample(
        n=n_coco_txt, random_state=42
    )
    coco_no_txt_sample = files_coco.loc[files_coco["subdir"] == "text_absence"].sample(
        n=n_coco_no_txt, random_state=42
    )

    # Select 50% - 50% text presence and text absence from Conceptual Captions dataset
    n_cc_no_txt = int(0.5 * n_cc)
    n_cc_txt = n_cc - n_cc_no_txt
    cc_txt_sample = files_cc.loc[files_cc["subdir"] == "text_presence"].sample(
        n=n_cc_txt, random_state=42
    )
    cc_no_txt_sample = files_cc.loc[files_cc["subdir"] == "text_absence"].sample(
        n=n_cc_no_txt, random_state=42
    )

    # ICDAR dataset sample
    icdar_sample = files_icdar.sample(n=n_icdar, random_state=42)

    # Concat all the samples to one DataFrame
    files_reg_img = pd.concat(
        [
            coco_txt_sample,
            coco_no_txt_sample,
            cc_txt_sample,
            cc_no_txt_sample,
            icdar_sample,
        ],
        ignore_index=True,
    )

    # Shuffle and split the dataframes to train, val and test
    perc_train = int(0.8 * n)
    perc_val = int(0.9 * n)

    train_memes, val_memes, test_memes = np.split(
        files_memes.sample(frac=1, random_state=42), [perc_train, perc_val]
    )

    train_reg_img, val_reg_img, test_reg_img = np.split(
        files_reg_img.sample(frac=1, random_state=42), [perc_train, perc_val]
    )

    # Create the train, val and test and shuffle the concatenated DataFrames
    train = pd.concat([train_memes, train_reg_img]).sample(
        frac=1,
        random_state=42,
        ignore_index=True,
    )
    val = pd.concat([val_memes, val_reg_img]).sample(
        frac=1,
        random_state=42,
        ignore_index=True,
    )
    test = pd.concat([test_memes, test_reg_img]).sample(
        frac=1,
        random_state=42,
        ignore_index=True,
    )

    return train, val, test


def get_files(directory: str, label: int) -> list:
    file = []
    for root, dirs, files in os.walk(directory):
        if not dirs:
            for file_ in files:
                file.append((os.path.join(root, file_), label, root.split("/")[-1]))

    return file


def make_dataset_dirs(
    meme_dir: str, coco_dir: str, icdar_dir: str, concep_capt_dir: str, out_dir: str
) -> None:
    """Function that creates the appropriate dataset directory to be used.
    The directory that will be created will have 3 subdirectories (train, valid, test)
    with the following structure:
    train  --memes -- image_macros
           |       |_ object_labelling
           |       |_ screenshots
           |       |_ text_out_of_image
           |
           |_regular_images -- coco -- with_text
                            |       |_ without_text
                            |
                            |_ icdar
                            |
                            |_ conceptual_captions -- with_text
                                                   |_ without_text
    etc.

    Variables:
        meme_dir: the path of the directory with the 4 meme categories already split
        coco_dir: the path of the COCO image dataset with subdirectories of text and no text images
        icdar_dir: the path of the ICDAR 2019 MLT dataset
        concep_capt_dir: the path of the Conceptual Captions directory
        out_dir: path to store the dataset
    """
    root_dir = os.path.join(out_dir, "meme_detection_dataset")
    regular_img_dir = os.path.join(root_dir, "regular_images")
    meme_img_dir = os.path.join(root_dir, "memes")

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(regular_img_dir):
        os.makedirs(regular_img_dir)
    if not os.path.exists(meme_img_dir):
        os.makedirs(meme_img_dir)

    # Copy meme image subdirectories to meme image directory
    meme_subdirs = [
        "screenshots",
        "image_macros",
        "object_labelling",
        "text_out_of_image",
    ]
    # for dir in meme_subdirs:
    #     shutil.copytree(os.path.join(meme_dir, dir), os.path.join(meme_img_dir, dir))

    # Find the meme category with the least amount of images and randomly
    # delete images from the other categories to balance the dataset
    smallest_dir = None
    min_images = float("inf")
    for root, dirs, files in os.walk(meme_img_dir):
        if not dirs:
            num_files = len(files)
            if min_images > num_files:
                min_images = num_files
                smallest_dir = root

    np.random.seed(42)

    for root, dirs, files in os.walk(meme_img_dir):
        if not dirs and not dirs == smallest_dir:
            # Number of files to delete
            num_files = len(files) - min_images
            indices = np.random.choice(len(files), num_files)
            for idx in indices:
                try:
                    os.remove(os.path.join(root, files[idx]))
                except:
                    pass
                    # print("Image already removed")

    shutil.copytree(coco_dir, os.path.join(regular_img_dir, "coco"))
    shutil.copytree(icdar_dir, os.path.join(regular_img_dir, "icdar"))
    shutil.copytree(
        concep_capt_dir, os.path.join(regular_img_dir, "conceptual_captions")
    )
