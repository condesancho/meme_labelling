import pandas as pd

import os
import shutil
import sys
import random

selection = int(input("Select 1 for cnn or 2 for vgg: "))
if selection == 1:
    feature_dir = "./cnn/"
elif selection == 2:
    feature_dir = "./vgg/"
else:
    sys.exit("Invalid input. Try again")

df = pd.read_csv(feature_dir + "predicted_labels.csv")

# Print the distribution of the images in the different categories
print(df["predicted_labels"].value_counts())

# # Split the images to separate folders according to the csv file
# for i in range(len(df.index)):
#     out_dir = feature_dir + "clusters/cluster" + str(df.predicted_labels[i])
#     shutil.copy(df.image[i].replace("varailop", "bill/extra"), out_dir)

# # Splits images with specific labels into folders
# for i in [1, 2, 4]:
#     temp_df = df[df["predicted_labels"] == i]
#     for j in temp_df.index:
#         out_dir = feature_dir + "clusters/cluster" + str(df.predicted_labels[j])
#         shutil.copy(df.image[j].replace("varailop", "bill/extra"), out_dir)

# # Takes random samples from the clusters and stores them in the samples folder
# for folder, subs, files in os.walk(feature_dir + "clusters"):
#     if not subs:  #  ignore the directory if it does have sub-directories
#         dest = "./samples/" + folder[-8:]
#         # select 100 random files
#         filenames = random.sample(os.listdir(folder), 100)

#         for fname in filenames:
#             srcpath = os.path.join(folder, fname)
#             dstpath = os.path.join(dest, fname)
#             shutil.copyfile(srcpath, dstpath)
