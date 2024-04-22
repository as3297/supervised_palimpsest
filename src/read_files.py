
import numpy as np
from pil_image_cube import ImageCubePILobject


class DataFromPILImageCube():
    """Factory for reading data from Pil MSI image"""
    def __init__(self, pil_msi_obj, max_vals):
        self.pil_msi_obj = pil_msi_obj
        self.max_vals = max_vals


    def contrast_stretch(self,msi_img):
        for band_idx in range(self.pil_msi_obj.nb_bands):
            msi_img[band_idx] = strech_contrast(msi_img[band_idx],self.max_vals[band_idx])
        return msi_img


    def convert_pil_to_array(self,pil_msi_obj):
        msi_img = conver_pil_msi_ims_to_array(pil_msi_obj.pil_msi_img,
                                              pil_msi_obj.width,
                                              pil_msi_obj.height,
                                              pil_msi_obj.nb_bands)
        return msi_img

class CoordsWindow():
    def __init__(self,pil_img_obj:ImageCubePILobject):
        """
        Factory for data window class parameters
        """
        self.width = pil_img_obj.width
        self.height = pil_img_obj.height
        self.pil_msi_img = pil_img_obj.pil_msi_img
        self.bands_list = pil_img_obj.band_list
        self.nb_bands = pil_img_obj.nb_bands


class BboxWindow(CoordsWindow):
    def __init__(self,bbox,pil_msi_obj):
        """
        
        :param bbox:
        """
        super().__init__(pil_msi_obj)
        self.bbox = bbox # left, upper, right, and lower pixel coordinate
        self.width = self.bbox[2] - self.bbox[0]
        self.height = self.bbox[3] - self.bbox[1]
        self.pil_msi_img = self.read_fragment()

    def read_fragment(self):
        """
        Read image extracted from bounding box
        :return: array with dim [nb_bands,nb_points]
        """
        fragment_pil_msi_img = []
        for im_band in self.pil_msi_img:
                fragment_pil_msi_img.append(im_band.crop(self.bbox))
        return fragment_pil_msi_img


class PointsWindow(CoordsWindow):
    def __init__(self, pil_msi_obj: ImageCubePILobject, points_coord):
        """

        :param bbox:
        """
        super().__init__(pil_msi_obj)
        self.width = None
        self.height = None
        self.points_coord = points_coord
        self.nb_points = len(self.points_coord)


class FullImageFromPILImageCube(DataFromPILImageCube):
    def __init__(self,pil_msi_obj,max_vals):
        """
        Read full msi image
        :param pil_msi_obj:
        :param max_vals:
        """
        super().__init__(pil_msi_obj, max_vals)
        self.unstretch_ims_img = self.convert_pil_msi_to_array(pil_msi_obj)
        self.ims_img = self.contrast_stretch(self.unstretch_ims_img)



class FragmentfromMSI_PIL(DataFromPILImageCube):
    def __init__(self,pil_msi_obj: list,max_vals: list,bbox):
        """
        Read points from image
        :param msi_img: list of PIL image objects of each band of MSI image
        :param max_vals_per_band:
        :param band_list: list of bands
        """
        super().__init__(pil_msi_obj, max_vals)
        self.bbox = bbox
        self.pil_msi_obj = BboxWindow(self.bbox,pil_msi_obj)
        self.unstretch_ims_img = self.convert_pil_to_array(self.pil_msi_obj)
        self.ims_img = self.contrast_stretch(self.unstretch_ims_img)



class PointsfromMSI_PIL(DataFromPILImageCube):
    def __init__(self,pil_msi_obj: list,max_vals: list,points_coord):
        """
        read points from image
        :param msi_img: list of PIL image objects of each band of MSI image
        :param max_vals_per_band:
        :param band_list: list of bands
        """
        super().__init__(pil_msi_obj, max_vals)
        self.points_coord = points_coord
        self.pil_msi_obj = PointsWindow(pil_msi_obj,self.points_coord)
        self.unstretch_ims_img = None
        self.unstretch_points = self.convert_pil_points_to_array(self.pil_msi_obj)
        self.points = self.contrast_stretch(self.unstretch_points)

    def convert_pil_points_to_array(self,pil_msi_obj):
        """
        Read values extracted from points
        :param points_coord:
        :return: array with dim [nb_bands,nb_points]
        """
        points = np.zeros([pil_msi_obj.nb_bands,pil_msi_obj.nb_points])
        for band_idx,im_band in enumerate(pil_msi_obj.pil_msi_img):
            for point_idx,coord in enumerate(self.points_coord):
                x,y = coord
                pixel_val = im_band.getpixel((x,y))
                points[band_idx,point_idx] = pixel_val
        return points


def strech_contrast(val,max_val):
    """Strech im by max value without oversaturated pixels
    max_val - int, bit depth"""
    val = np.clip(val,a_max=max_val,a_min=0.)
    val= val/max_val
    return val



def conver_pil_msi_ims_to_array(pil_msi_img,width,height,nb_bands):
    """Convert PIL image cube into array image cube"""

    msi_ims = np.zeros([nb_bands,height,width])
    for band_idx, im_band in enumerate(pil_msi_img):
        im = np.array(im_band)
        msi_ims[band_idx] = im
    return msi_ims






