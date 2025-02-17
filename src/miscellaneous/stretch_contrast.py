import cv2
import numpy as np
import os
import shutil


pal_path = "d:\Verona_msXL"
folio_name = "msXL_315r_b"
fnames = ["undertext", "not_undertext", "spectralon"]
# Check if 'mask_copy' folder exists, if not create it
mask_copy_path = os.path.join(pal_path, folio_name, "mask_copy")
if not os.path.exists(mask_copy_path):
    os.makedirs(mask_copy_path)
# Read the grayscale image
for fname in fnames:
    fname = f"{folio_name}-{fname}_black.png"
    image_path = os.path.join(pal_path,folio_name,"mask",fname)
    shutil.copy(os.path.join(pal_path,folio_name,"mask",fname),os.path.join(mask_copy_path,fname))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Stretch contrast to make image values between 0 and 1
    image_stretched = cv2.normalize(image.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    # Save the processed image as PNG
    cv2.imwrite(image_path, (image_stretched * 255).astype(np.uint8))
    