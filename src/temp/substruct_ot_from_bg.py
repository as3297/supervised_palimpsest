import cv2
import numpy as np
from skimage import io
import os

# Read the image
folio_dir = r"c:\Data\PhD\palimpsest\Victor_data\Paris_Coislin"
palimpsest_name = r"Par_coislin_393_054r"
class_name = "bg_lines"
image_path = os.path.join(folio_dir,palimpsest_name,"mask",f"{palimpsest_name}-{class_name}_black.png")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully loaded
if image is None:
    raise FileNotFoundError(f"Unable to read the image at path: {image_path}")

# Read the second image
second_image_path = os.path.join(folio_dir,palimpsest_name,"mask",f"{palimpsest_name}-overtext_black.png")
second_image = cv2.imread(second_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the second image was successfully loaded
if second_image is None:
    raise FileNotFoundError(f"Unable to read the image at path: {second_image_path}")

if np.amax(image)>1:
    image = image/255
if np.amax(second_image)>1:
    second_image = second_image/255
# Invert the image
image[second_image==0] = 1
image[image<0.1] = 0
image[image>0.0] = 1

image = image.astype("bool")
# Save the subtracted image if needed
subtracted_output_path = os.path.join(folio_dir,palimpsest_name,"mask",f"{palimpsest_name}-{class_name}_ot_subtracted_black.png")
io.imsave(subtracted_output_path,image)


