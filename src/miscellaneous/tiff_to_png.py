import numpy as np

from src.pil_image_cube import ImageCubePILobject
from src.msi_data_as_array import FullImageFromPILImageCube
from PIL import Image
import os

from src.util import read_band_list

folios = ["msXL_335v_b", "msXL_335v_b", "msXL_315v_b", "msXL_318r_b",
          "msXL_318v_b", "msXL_319r_b", "msXL_319v_b", "msXL_322r_b",
          "msXL_322v_b", "msXL_323r_b", "msXL_334r_b", "msXL_334v_b",
          "msXL_344r_b", "msXL_344v_b", "msXL_315r_b"]

main_data_dir = r"/home/anna/projects/palimpsests/Verona_msXL"
modalities = ["M"]
output_dir = os.path.join(main_data_dir, "png_images_standardized")

os.makedirs(output_dir, exist_ok=True)
band_list = read_band_list(os.path.join(main_data_dir,"band_list.txt"), modalities)

for folio in folios:
    pil_msi_obj = ImageCubePILobject(main_data_dir, folio, band_list, 0)
    msi_img = FullImageFromPILImageCube(pil_msi_obj).ims_img
    band_list = pil_msi_obj.band_list
    new_folio_path = os.path.join(output_dir, folio)
    os.makedirs(new_folio_path, exist_ok=True)
    for i, band in enumerate(band_list):
        img_band = np.round(msi_img[:, :, i]*255)
        image_band = Image.fromarray(img_band.astype(np.uint8))
        image_band = image_band.convert("L")
        output_path = os.path.join(new_folio_path, f"{folio}-{band}.png")
        image_band.save(output_path)
