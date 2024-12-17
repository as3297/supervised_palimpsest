import os.path

import cv2
import numpy as np

# Read the image
working_dir = r"C:\Data\PhD\palimpsest\Victor_data\Paris_Coislin\Par_coislin_393_054r\miscellaneous\Naive_Bayesian"
fname = "Par_coislin_393_054r_NB_undertext_bg_lines.png"
image_path = os.path.join(
    working_dir, fname)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully loaded
if image is None:
    raise FileNotFoundError(f"Unable to read the image at path: {image_path}")
if np.amax(image)>1:
    image = image/255

# Read the second image
ot_image_path = "C:\\Data\\PhD\\palimpsest\\Victor_data\\Paris_Coislin\\Par_coislin_393_054r\\mask\\Par_coislin_393_054r-overtext_black.png"
ot_image = cv2.imread(ot_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the second image was successfully loaded
if ot_image is None:
    raise FileNotFoundError(f"Unable to read the image at path: {ot_image_path}")
if np.amax(ot_image)>1:
    ot_image = ot_image/255
# Invert the image
inverted_ot_image = 1 - ot_image
# Subtract the second image from the first
subtracted_image = image - inverted_ot_image
# Save the subtracted image if needed
subtracted_image = (255*subtracted_image).astype("uint8")
subtracted_output_path = os.path.join(working_dir,fname[:-4]+"_subtracted.png" )
cv2.imwrite(subtracted_output_path, subtracted_image)


