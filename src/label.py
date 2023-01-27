import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os
import shutil

df = pd.read_json("./tsne/points.json")
points = pd.DataFrame(df["point"].to_list(), columns=["x", "y"])

kmeans = KMeans(n_clusters=5, n_init="auto").fit(points.to_numpy())
plt.scatter(points.x, points.y, c=kmeans.labels_)
plt.show()

# Split the images to separate folders according to the kmeans algorithm
# for i in range(len(df.index)):
#     out_dir = "./clusters/cluster" + str(kmeans.labels_[i])
#     shutil.copy(df.path[i], out_dir)
