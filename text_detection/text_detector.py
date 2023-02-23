"""
This is a file that uses TextFuseNet implemetation to detect if there is text on an image
"""

# @inproceedings{ijcai2020-72,
#     title={TextFuseNet: Scene Text Detection with Richer Fused Features},
#     author={Ye, Jian and Chen, Zhe and Liu, Juhua and Du, Bo},
#     booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, {IJCAI-20}},
#     publisher={International Joint Conferences on Artificial Intelligence Organization},
#     pages={516--522},
#     year={2020}
# }

import argparse
import numpy as np
import os
from TextFuseNet.detectron2.config import get_cfg
from text_detection.predictor import VisualizationDemo
import cv2
from PIL import Image
import time

import pandas as pd


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set model
    cfg.MODEL.WEIGHTS = args.weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


def get_parser():
    base_dir = "./TextFuseNet"
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default=base_dir + "/configs/ocr/icdar2013_101_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--weights",
        default=base_dir + "/out_dir_r101/icdar2013_model/model_ic13_r101.pth",
        metavar="pth",
        help="the model used to inference",
    )

    parser.add_argument(
        "--input",
        default=base_dir + "/input_images/*.jpg",
        nargs="+",
        help="the folder of icdar2013 test images",
    )

    parser.add_argument(
        "--output",
        default=base_dir + "/test_icdar2013/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


# Function that returns True if there is text on the image or False if there is not
def has_text(img_path):
    img = cv2.imread(img_path)
    if img.shape[-1] == 3:
        # Find text bounding boxes
        prediction, _, _ = detection_demo.run_on_image(img)
        classes = prediction["instances"].pred_classes
        # If text is detected in the image
        if len(classes) == 0:
            del classes
            del prediction
            return False
        else:
            del classes
            del prediction
            return True


args = get_parser().parse_args()
cfg = setup_cfg(args)
detection_demo = VisualizationDemo(cfg)
imgdir = "./data/images/"
files = [
    os.path.join(imgdir, x) for x in os.listdir(imgdir) if (".jpg" in x or ".png" in x)
]
n = len(files)
start = time.time()
counter = 0
df = pd.DataFrame(columns=["image", "text"])

for i, file in enumerate(files):
    df.loc[len(df.index)] = [file, has_text(file)]
    elapsed = time.time() - start
    print(
        f"\r{i + 1}/{n}: {file} {elapsed / 3600: 1.3f} h, ETA: {elapsed / (i + 1) * (n - i - 1) / 3600: 1.3f} h",
        end="",
    )

df.to_csv("./text_detection/text_presence.csv", index=True)
