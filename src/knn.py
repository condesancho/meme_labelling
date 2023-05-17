from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import sys

selection = int(
    input(
        "Select 1 for CNN, 2 for VGG, 3 for ResNet, 4 for EfficientNet or 5 for ViT: "
    )
)
if selection == 1:
    feature_dir = "../models/cnn_feat_ex/"
elif selection == 2:
    feature_dir = "../models/vgg_feat_ex/"
elif selection == 3:
    feature_dir = "../models/resnet_feat_ex/"
elif selection == 4:
    feature_dir = "../models/efficientnet_feat_ex/"
elif selection == 5:
    feature_dir = "../models/vit_feat_ex/"
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
