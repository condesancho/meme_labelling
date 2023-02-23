import pandas as pd

import os
import shutil
import sys
import random

selection = int(
    input(
        "Select 1 for cnn with feature extraction,\n2 for vgg with feature extraction, \n3 for cnn, \n4 for vgg: "
    )
)
if selection == 1:
    label_dir = "../models/cnn_feat_ex/"
elif selection == 2:
    label_dir = "../models/vgg_feat_ex/"
elif selection == 3:
    label_dir = "../models/cnn/"
elif selection == 4:
    label_dir = "../models/vgg/"
else:
    sys.exit("Invalid input. Try again")

df = pd.read_csv(label_dir + "predicted_labels.csv")
print(df.shape)

text_presence = pd.read_csv("../text_detection/text_presence.csv")
no_text = text_presence[text_presence.text == False].copy()

# Drop rows where the image doesn't contain text
no_text.replace(
    to_replace="./data",
    value="/home/varailop/meme_labelling/data",
    regex=True,
    inplace=True,
)
df = df[df.loc[:, "image"].isin(no_text.loc[:, "image"]) == False]

# Print the distribution of the images in the different categories
print(df["predicted_labels"].value_counts())

# Equally sample images of each label from the dataframe and copy them
df_samples = df.groupby("predicted_labels").sample(n=500, ignore_index=True)
for i in range(len(df_samples.index)):
    out_dir = label_dir + "samples/cluster" + str(df_samples.predicted_labels[i])
    shutil.copy(df.image[i].replace("varailop", "bill/extra"), out_dir)

# # Split the images to separate folders according to the csv file
# for i in range(len(df.index)):
#     out_dir = label_dir + "clusters/cluster" + str(df.predicted_labels[i])
#     shutil.copy(df.image[i].replace("varailop", "bill/extra"), out_dir)

# # Splits images with specific labels into folders
# for i in [1, 2, 4]:
#     temp_df = df[df["predicted_labels"] == i]
#     for j in temp_df.index:
#         out_dir = label_dir + "clusters/cluster" + str(df.predicted_labels[j])
#         shutil.copy(df.image[j].replace("varailop", "bill/extra"), out_dir)

# # Takes random samples from the clusters and stores them in the samples folder
# for folder, subs, files in os.walk(label_dir + "clusters"):
#     if not subs:  #  ignore the directory if it does have sub-directories
#         dest = "./samples/" + folder[-8:]
#         # select 100 random files
#         filenames = random.sample(os.listdir(folder), 100)

#         for fname in filenames:
#             srcpath = os.path.join(folder, fname)
#             dstpath = os.path.join(dest, fname)
#             shutil.copyfile(srcpath, dstpath)
