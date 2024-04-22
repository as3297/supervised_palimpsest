from PIL import Image
import os


class ImageCubePILobject:
    def __init__(self,image_dir,band_list,rotate_angle):
        """
        :param image_dir: directory with tif image of palimpsest
        :param band_list: list of bands
        :param coord: (left, upper, right, lower) tuple of bounding box coordinates
        """
        self.image_dir = image_dir
        if len(band_list)==0:
            for fname in os.listdir(self.image_dir):
                if ".tif" in fname:
                    band_list.append(fname[:-4])
        self.band_list = band_list
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
            fpath = os.path.join(self.image_dir, band_name + ".tif")
            im = self.read_image_object(fpath)
            msi_img.append(im)
        return msi_img

    def close_all_images(self):
        for band_obj in self.pil_msi_img:
            band_obj.close()

class ThumbnailMSI_PIL(ImageCubePILobject):
    def __init__(self,image_dir,band_list,rotate_angle,scale_ratio):
        """Resize msi_img_obj"""
        super().__init__(image_dir,band_list,rotate_angle)
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