from PIL import Image
import os
from read_pixel_coord import CoordfromMask
from util import sublist_of_bands

class ImageCubePILobject:
    def __init__(self,image_dir,folio_name,band_list,rotate_angle):
        """
        Read MSI image cube as a list of PIL images from a dir with stored image bands as tif images.
        The folder should contain only images of actual bands.
        The til file naming format is "folio name"-"band name"_"band index"_F.tif, e.g. msXL_315r_b-M0365UV_01_F.tif,
        where msXL_315r_b - folio name, M0365UV - band name, 01 - band index.
        :param image_dir: directory with tif image of palimpsest
        :param folio_name: name of the folio
        :param band_list: list of bands
        :param coord: (left, upper, right, lower) tuple of bounding box coordinates
        """
        self.image_dir = image_dir
        self.folio_name = folio_name
        if len(band_list)==0:
            for fname in os.listdir(self.image_dir):
                if ".tif" in fname:
                    band_list.append(fname[:-4])
        self.band_list = band_list
        self.rotate_angle = rotate_angle
        self.pil_msi_img = self.read_msi_image_object()
        self.width, self.height = self.pil_msi_img[0].size
        self.nb_bands = len(self.band_list)
        self.spectralon_mask_path = os.path.join(self.image_dir, "mask", self.folio_name + "-" + "spectralon.png")
        self.spectralon_coords = CoordfromMask(self.spectralon_mask_path, self.rotate_angle).coords



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
            im = self.read_image_object(fpath)
            msi_img.append(im)
        return msi_img

    def close_all_images(self):
        for band_obj in self.pil_msi_img:
            band_obj.close()

class ThumbnailMSI_PIL(ImageCubePILobject):
    def __init__(self,image_dir,folio_name,band_list,rotate_angle,scale_ratio):
        """Resize msi_img_obj"""
        super().__init__(image_dir,folio_name,band_list,rotate_angle)
        self.width = self.width // scale_ratio
        self.height = self.height // scale_ratio
        self.msi_img_thumbnail = self.thumbnail_msi()

    def thumbnail_msi(self):
        """Resize msi image pil object"""
        thumbnail_msi_img = []
        for im_band_pil in self.pil_msi_img:
            im = im_band_pil.resize((self.width,self.height))
            thumbnail_msi_img.append(im)
        return thumbnail_msi_img

class MB_MSI_PIL(ImageCubePILobject):
    def __init__(self, image_dir, folio_name, band_list, rotate_angle):
        """MB msi_img_obj"""

        self.band_list = sublist_of_bands(band_list,"M")

        super().__init__(image_dir, folio_name, band_list, rotate_angle)
