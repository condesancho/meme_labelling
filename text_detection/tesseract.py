import cv2
import pytesseract
import os
import time
import pandas as pd

imgdir = "../data/images/"
files = [
    os.path.join(imgdir, x) for x in os.listdir(imgdir) if (".jpg" in x or ".png" in x)
]
n = len(files)
start = time.time()
counter = 0
df = pd.DataFrame(columns=["image", "text"])

for i, file in enumerate(files):
    image = cv2.imread(file)
    text = pytesseract.image_to_string(image)
    if text.strip():
        has_text = True
    else:
        has_text = False
    df.loc[len(df.index)] = [file, has_text]
    elapsed = time.time() - start
    print(
        f"\r{i + 1}/{n}: {file} {elapsed / 3600: 1.3f} h, ETA: {elapsed / (i + 1) * (n - i - 1) / 3600: 1.3f} h",
        end="",
    )

df.to_csv("./text_presence.csv", index=True)
