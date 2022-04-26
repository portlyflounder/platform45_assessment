"""A python script that creates a collage of annoated dog images."""

import cv2
import math
import matplotlib.pyplot as plt
from pathlib import Path

GOOD_ANNOATED_IMAGE_DIR = "data/annotated_regions"
IMAGE_RESIZE_HEIGHT = 224
IMAGE_RESIZE_WIDTH = 224
FIGURE_SIZE = (8, 8)
OUTPUT_FILEPATH = "doggy_collage.png"

def run():
    image_filepaths = [image_filepath for image_filepath in Path(GOOD_ANNOATED_IMAGE_DIR).iterdir()]
    num_images = len(image_filepaths)
    num_cols = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=FIGURE_SIZE, gridspec_kw = {'wspace':0, 'hspace':0})
    for filepath, ax in zip(image_filepaths, axes.flatten()):
        image = cv2.imread(str(filepath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_RESIZE_HEIGHT, IMAGE_RESIZE_WIDTH))
        ax.imshow(image)
    
    for ax in axes.flatten():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(OUTPUT_FILEPATH, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    run()