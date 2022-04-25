"""A python script that extracts annotated dog regions from images."""

import cv2
import numpy as np
from pathlib import Path

ANNOTATED_IMAGE_DIR = "data/annotations"
RAW_IMAGE_DIR = "data/raw"

def run():
    for anno_image_filepath in Path(ANNOTATED_IMAGE_DIR).iterdir():
        anno_image_name = anno_image_filepath.name
        raw_image_filepath = Path(RAW_IMAGE_DIR).joinpath(anno_image_name).with_suffix(".jpg")

        if not raw_image_filepath.exists():
            raise FileNotFoundError(f"The raw image '{raw_image_filepath}' does not exist for the annotated image {anno_image_filepath}.")
        
        anno_image_gray = cv2.imread(str(anno_image_filepath), cv2.IMREAD_GRAYSCALE)
        raw_image = cv2.imread(str(raw_image_filepath))

        annotated_region_mask = (anno_image_gray < 255).astype(np.uint8)
        annotated_region = raw_image * np.stack([annotated_region_mask, annotated_region_mask, annotated_region_mask], axis=-1)
        
        
    
if __name__ == "__main__":
    run()