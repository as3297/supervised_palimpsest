from PIL import Image
import os
from src.util import read_band_list


class ImageCubePILobject:
    def __init__(self,folio_dir,folio_name,modalities,rotate_angle):
        """
        Read MSI image cube as a list of PIL images from a dir with stored image bands as tif images.
        The folder should contain only images of actual bands.
        The til file naming format is "folio name"-"band name"_"band index"_F.tif, e.g. msXL_315r_b-M0365UV_01_F.tif,
        where msXL_315r_b - folio name, M0365UV - band name, 01 - band index.
        :param image_dir: directory with tif image of palimpsest
        :param folio_name: name of the folio
        :param modalities: list of modalities
        :param coord: (left, upper, right, lower) tuple of bounding box coordinates
        """
        self.folio_dir = folio_dir
        self.image_dir = os.path.join(folio_dir, folio_name)
        self.folio_name = folio_name
        self.band_list = read_band_list(os.path.join(self.folio_dir,"band_list.txt"), modalities)
        self.rotate_angle = rotate_angle
        self.pil_msi_img = self.read_msi_image_object()
        self.width, self.height = self.pil_msi_img[0].size
        self.nb_bands = len(self.band_list)

    def read_image_object(self,path):
        im = Image.open(path)
        if self.rotate_angle>0:
            rotation = eval("Image.ROTATE_{}".format(self.rotate_angle))
            im = im.transpose(rotation)
        return im


    def read_msi_image_object(self):
        """Creates a list of PIL objects that correspond to each band of the image"""
        msi_img = []
        for idx, band_name in enumerate(self.band_list):
            fpath = os.path.join(self.image_dir, self.folio_name + "-" + band_name + ".tif")
            if not os.path.exists(fpath):
                fpath = os.path.join(self.image_dir, self.folio_name + "+" + band_name + ".tif")
            im = self.read_image_object(fpath)
            msi_img.append(im)
        return msi_img

    def close_all_images(self):
        for band_obj in self.pil_msi_img:
            band_obj.close()


class ImageCubeObject:
    def __init__(self,folio_dir,folio_name,modalities,rotate_angle):
        """
        Read MSI image cube as a list of PIL images from a dir with stored image bands as tif images.
        The folder should contain only images of actual bands.
        The til file naming format is "folio name"-"band name"_"band index"_F.tif, e.g. msXL_315r_b-M0365UV_01_F.tif,
        where msXL_315r_b - folio name, M0365UV - band name, 01 - band index.
        :param image_dir: directory with tif image of palimpsest
        :param folio_name: name of the folio
        :param modalities: list of modalities
        :param coord: (left, upper, right, lower) tuple of bounding box coordinates
        """
        self.folio_dir = folio_dir
        self.image_dir = os.path.join(folio_dir, folio_name)
        self.folio_name = folio_name
        self.band_list = read_band_list(os.path.join(self.folio_dir,"band_list.txt"), modalities)
        self.rotate_angle = rotate_angle
        self.nb_bands = len(self.band_list)





