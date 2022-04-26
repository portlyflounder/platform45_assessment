"""A python script that creates a collage of annoated dog images."""

import cv2
import math
import matplotlib.pyplot as plt
from pathlib import Path

# The directory that contains the acceptable annotated images.
GOOD_ANNOATED_IMAGE_DIR = "data/annotated_regions"

# The image height to resize each image to.
IMAGE_RESIZE_HEIGHT = 224

# The image width to resize each image to.
IMAGE_RESIZE_WIDTH = 224

# The figure/collage size.
FIGURE_SIZE = (8, 8)

# The filepath to save the doggy collage image to.
OUTPUT_FILEPATH = "data/doggy_collage.png"

def run():
    """Creates a collage of doggy images from a directory."""

    # Store each image filepath in a list.
    image_filepaths = [image_filepath for image_filepath in Path(GOOD_ANNOATED_IMAGE_DIR).iterdir()]

    # Determine the number of rows and columns based on the number of images. 
    # This ensures that the collage image is as square as possible.
    num_images = len(image_filepaths)
    num_cols = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)

    # Create subplots to render the images on.
    _, axes = plt.subplots(num_rows, num_cols, figsize=FIGURE_SIZE, gridspec_kw = {'wspace':0, 'hspace':0})

    # Load each image and render it on a dedicated subplot.
    for filepath, ax in zip(image_filepaths, axes.flatten()):
        image = cv2.imread(str(filepath))
        # Convert image to RGB since opencv loads images as BGR.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize and render the image.
        image = cv2.resize(image, (IMAGE_RESIZE_HEIGHT, IMAGE_RESIZE_WIDTH))
        ax.imshow(image)
    
    # Disable axis annotations for each subplot.
    for ax in axes.flatten():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Save the collage image to file.
    plt.savefig(OUTPUT_FILEPATH, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    run()