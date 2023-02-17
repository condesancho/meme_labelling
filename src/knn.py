from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import sys

selection = int(input("Select 1 for cnn or 2 for vgg: "))
if selection == 1:
    feature_dir = "./cnn/"
elif selection == 2:
    feature_dir = "./vgg/"
else:
    sys.exit("Invalid input. Try again")

labelled_data = pd.read_csv(feature_dir + "labelled_features.csv")
unlabelled_data = pd.read_csv(feature_dir + "unlabelled_features.csv")

knn = KNeighborsClassifier(n_neighbors=10)

X = labelled_data.iloc[:, :-1]
y = labelled_data.iloc[:, -1]

knn.fit(X, y)

x_unlabelled = unlabelled_data.iloc[:, :-1]
unlabelled_filepaths = unlabelled_data.iloc[:, -1]

predicted_labels = knn.predict(x_unlabelled)

df = pd.DataFrame()

df["predicted_labels"] = predicted_labels
df = df.join(unlabelled_filepaths)
df.to_csv(feature_dir + "predicted_labels.csv", index=True)
