"""A python script that extracts annotated dog regions from images."""

import cv2
import json
import numpy as np
from pathlib import Path

# The json file that contains the reviewed annotations.
JSON_FILE = "data/reviewed_annotations.json"

# The directory to store the annotated region images in.
ANNOTATED_REGIONS_OUTPUT_DIR = "data/annotated_regions"

def run():
    """Extracts the annoated doggy regions from each image and saves it to file."""

    # Load the json file containing the reviewed annotations.
    with open(JSON_FILE, "r") as json_file:
        reviewed_image_data = json.loads(json_file.read())

    # Iterate over each annotation entry. We only extract annotated doggy 
    # regions if the annotation is marked as "acceptable".
    for metadata in reviewed_image_data.values():
        acceptable_annotation = metadata["acceptable"]
        if acceptable_annotation:
            filepaths = metadata["filepaths"]
            anno_image_filepath = filepaths["annotation"]
            raw_image_filepath = filepaths["raw"]

            # Load the annotated image and raw image. We convert the annotated image to 
            # grayscale so that we can threshold it to create a binary mask.
            anno_image_gray = cv2.imread(str(anno_image_filepath), cv2.IMREAD_GRAYSCALE)
            raw_image = cv2.imread(str(raw_image_filepath))

            # Threshold the annotated image. We know that anything below 255 is part of the annotated region.
            annotated_region_mask = (anno_image_gray < 255).astype(np.uint8)

            # Apply the annotated region mask to the raw image.
            # We stack the annotated mask to create a 3 channel image (i.e. mocking an RGB image).
            annotated_region = raw_image * np.stack([annotated_region_mask] * 3, axis=-1)
            
            # Create the filepath of the annoated region image and save to file.
            annotated_region_filepath = Path(ANNOTATED_REGIONS_OUTPUT_DIR).joinpath(Path(anno_image_filepath).name).with_suffix(".jpg")
            cv2.imwrite(str(annotated_region_filepath), annotated_region)
    
if __name__ == "__main__":
    run()