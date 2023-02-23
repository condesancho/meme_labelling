"""
File that reads the csv file and copies the images with no text to a new directory
"""

import pandas as pd

import shutil

df = pd.read_csv("./text_presence.csv")

# Print the distribution of the images in the different categories
print(df["text"].value_counts())

# Split the images to separate folders according to the csv file
for i in range(len(df.index)):
    out_dir = "./no_text/"
    if not df.text[i]:
        shutil.copy(df.image[i].replace("./data", "../data"), out_dir)
