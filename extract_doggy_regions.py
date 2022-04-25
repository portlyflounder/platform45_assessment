"""A python script that extracts annotated dog regions from images."""

import cv2
import json
import numpy as np
from pathlib import Path

JSON_FILE = "reviewed_labels.json"
ANNOTATED_REGIONS_OUTPUT_DIR = "data/annotated_regions"

def run():
    with open(JSON_FILE, "r") as json_file:
        reviewed_image_data = json.loads(json_file.read())
    for metadata in reviewed_image_data.values():
        acceptable_annotation = metadata["acceptable"]
        if acceptable_annotation:
            filepaths = metadata["filepaths"]
            anno_image_filepath = filepaths["annotation"]
            raw_image_filepath = filepaths["raw"]

            anno_image_gray = cv2.imread(str(anno_image_filepath), cv2.IMREAD_GRAYSCALE)
            raw_image = cv2.imread(str(raw_image_filepath))

            annotated_region_mask = (anno_image_gray < 255).astype(np.uint8)
            annotated_region = raw_image * np.stack([annotated_region_mask] * 3, axis=-1)
            annotated_region_filepath = Path(ANNOTATED_REGIONS_OUTPUT_DIR).joinpath(Path(anno_image_filepath).name).with_suffix(".jpg")
            cv2.imwrite(str(annotated_region_filepath), annotated_region)
    
if __name__ == "__main__":
    run()