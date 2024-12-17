import numpy as np
from pil_image_cube import ImageCubePILobject
from pixel_coord import ClassCoord
import os
from util import read_json
import copy

osp = os.path.join
class NormalizingGray():
    def __init__(self, pil_msi_obj: ImageCubePILobject):
        """Reads means gray value of specralon target for reflectance band or max values for flourescent band"""
        self.pil_msi_obj = pil_msi_obj
        self.spectralon_coords = self.read_spectralon_coords()
        self.max_values_dict = read_json(osp(self.pil_msi_obj.folio_dir,"bands_max_val.json"))
        self.max_values = self.make_array_of_normalizing_values()

    def read_spectralon_coords(self):
        self.spectralon_mask_path = osp(self.pil_msi_obj.image_dir, "mask", self.pil_msi_obj.folio_name + "-" + "spectralon_black.png")
        s = ClassCoord(self.spectralon_mask_path, self.pil_msi_obj.rotate_angle)
        return s.coords

    def read_spectralon_value(self,band_idx):
        """Read spectralon value for one band"""
        im_band = self.pil_msi_obj.pil_msi_img[band_idx]
        points_per_band = list(map(im_band.getpixel, self.spectralon_coords))
        mean_grey_val = np.mean(points_per_band)
        return mean_grey_val

    def make_array_of_normalizing_values(self):
        band_list = self.pil_msi_obj.band_list
        max_vals = np.zeros((len(band_list)))
        for band_idx,band_name in enumerate(band_list):
            if band_name[0].lower()=="m":
                max_val = self.read_spectralon_value(band_idx)
            elif band_name[0].lower()=="w":
                max_val = self.max_values_dict[band_name]
            max_vals[band_idx]=max_val
        return max_vals


class DataFromPILImageCube():
    """Class for reading data from Pil MSI image"""
    def __init__(self, pil_msi_obj:ImageCubePILobject):
        self.pil_msi_obj = pil_msi_obj
        norm_val_ob = NormalizingGray(self.pil_msi_obj)
        self.max_vals = norm_val_ob.max_values

    def standartize(self,msi_img):
        """
        Standartize
        """
        for band_idx in range(self.pil_msi_obj.nb_bands):
            msi_img[band_idx] = standartize(msi_img[band_idx],self.max_vals[band_idx])
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
        Factory for data ROI class parameters
        """
        self.width = pil_img_obj.width
        self.height = pil_img_obj.height
        self.pil_msi_img = pil_img_obj.pil_msi_img
        self.bands_list = pil_img_obj.band_list
        self.nb_bands = pil_img_obj.nb_bands


class BboxWindow(CoordsWindow):
    def __init__(self,bbox,pil_msi_obj):
        """
        ROI class in shape of box
        :param bbox:  [left, upper, right, and lower] pixel coordinate
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
        ROI class as individual points
        :param bbox:
        """
        super().__init__(pil_msi_obj)
        self.width = None
        self.height = None
        self.points_coord = points_coord
        self.nb_points = len(self.points_coord)

class FullImageFromPILImageCube(DataFromPILImageCube):
    def __init__(self,pil_msi_obj):
        """
        Read full msi image
        :param pil_msi_obj:
        """
        super().__init__(pil_msi_obj)
        self.unstretch_ims_img = self.convert_pil_to_array(pil_msi_obj)
        im = copy.deepcopy(self.unstretch_ims_img)
        ims_img = self.standartize(im)
        self.ims_img = np.transpose(ims_img, axes=[2, 0, 1])

class FragmentfromMSI_PIL(DataFromPILImageCube):
    def __init__(self,pil_msi_obj:ImageCubePILobject,bbox):
        """
        Read points from image
        :param msi_img: list of PIL image objects of each band of MSI image
        :param bbox: [left, upper, right, and lower]

        """
        super().__init__(pil_msi_obj)
        self.bbox = bbox
        self.pil_msi_obj = BboxWindow(self.bbox,pil_msi_obj)
        self.unstretch_ims_img = self.convert_pil_to_array(self.pil_msi_obj)
        ims_img = self.standartize(self.unstretch_ims_img)
        self.ims_img = np.transpose(ims_img, axes=[2, 0, 1])



class PointsfromMSI_PIL(DataFromPILImageCube):
    def __init__(self,pil_msi_obj: ImageCubePILobject,points_coord):
        """
        read points from image
        :param msi_img: list of PIL image objects of each band of MSI image
        :param max_vals_per_band:
        :param points_coords: list of coordinates
        """
        super().__init__(pil_msi_obj)
        self.points_coord = points_coord
        self.pil_msi_obj = PointsWindow(pil_msi_obj,self.points_coord)
        self.unstretch_ims_img = None
        self.unstretch_points = self.convert_pil_points_to_array(self.pil_msi_obj)
        points = self.standartize(self.unstretch_points)
        self.points = np.transpose(points,[1,0])


    def convert_pil_points_to_array(self,pil_msi_obj):
        """
        Read values extracted from points
        :param points_coord:
        :return: array with dim [nb_bands,nb_points]
        """
        points = np.zeros([pil_msi_obj.nb_bands,pil_msi_obj.nb_points])
        for band_idx,im_band in enumerate(pil_msi_obj.pil_msi_img):
            points_per_band = list(map(im_band.getpixel,self.points_coord))
            points[band_idx] = points_per_band
        return points

class PointfromMSI_PIL(DataFromPILImageCube):
    def __init__(self,pil_msi_obj: ImageCubePILobject,point_coord):
        """
        read points from image
        :param msi_img: list of PIL image objects of each band of MSI image
        :param max_vals_per_band:
        :param points_coords: list of coordinates
        """
        super().__init__(pil_msi_obj)
        self.point_coord = point_coord
        self.unstretch_ims_img = None
        self.unstretch_point = self.convert_pil_points_to_array(self.pil_msi_obj)
        self.point = self.standartize(self.unstretch_point)


    def convert_pil_points_to_array(self,pil_msi_obj):
        """
        Read values extracted from points
        :param points_coord:
        :return: array with dim [nb_bands,nb_points]
        """
        point = np.zeros([pil_msi_obj.nb_bands,])
        for band_idx,im_band in enumerate(pil_msi_obj.pil_msi_img):
            point_per_band = im_band.getpixel(self.point_coord)
            point[band_idx] = point_per_band
        return point

class PointsfromRatio(DataFromPILImageCube):
    def __init__(self, pil_msi_obj: ImageCubePILobject, points_coord):
        """
        read points from image
        :param msi_img: list of PIL image objects of each band of MSI image
        :param max_vals_per_band:
        :param points_coords: list of coordinates
        """
        super().__init__(pil_msi_obj)
        self.points_coord = points_coord
        self.pil_msi_obj = PointsWindow(pil_msi_obj, self.points_coord)
        self.unstretch_ims_img = None

        self.points_ratio_W420B47_W385UVB = self.convert_pil_points_to_array_ratio_W420B47_W385UVB(self.pil_msi_obj).reshape([-1,1])
        self.points_ratio_W365UVP_W385UVB = self.convert_pil_points_to_array_ratio_W365UVP_W385UVB(self.pil_msi_obj).reshape([-1,1])
        print("ratio shape,", self.points_ratio_W365UVP_W385UVB.shape)

    def convert_pil_points_to_array_ratio_W365UVP_W385UVB(self,pil_msi_obj):
        """
        Read values extracted from points
        :param points_coord:
        :return: array with dim [nb_bands,nb_points]
        """
        for band_idx, im_band in enumerate(pil_msi_obj.bands_list):
            if im_band[0].lower() == "w":
                if im_band == "W365UVP_27_F":
                    points_W365UVP = list(map(pil_msi_obj.pil_msi_img[band_idx].getpixel, self.points_coord))
                if im_band == "W385UVB_21_F":
                    points_W385UVB = list(map(pil_msi_obj.pil_msi_img[band_idx].getpixel,self.points_coord))
        points_ratio = np.array(points_W365UVP)/(np.array(points_W385UVB)+1e-06)
        return points_ratio

    def convert_pil_points_to_array_ratio_W420B47_W385UVB(self,pil_msi_obj):
        """
        Read values extracted from points
        :param points_coord:
        :return: array with dim [nb_bands,nb_points]
        """
        for band_idx, im_band in enumerate(pil_msi_obj.bands_list):
            if im_band[0].lower() == "w":
                if im_band == "W420B47_42_F":
                    points_W420B47 = list(map(pil_msi_obj.pil_msi_img[band_idx].getpixel, self.points_coord))
                if im_band == "W385UVB_21_F":
                    points_W385UVB = list(map(pil_msi_obj.pil_msi_img[band_idx].getpixel,self.points_coord))
        points_ratio = np.array(points_W420B47)/(np.array(points_W385UVB)+1e-06)
        return points_ratio

class PointsfromBand(DataFromPILImageCube):
    def __init__(self, pil_msi_obj: ImageCubePILobject, points_coord, band_name):
        """
        read points from image
        :param msi_img: list of PIL image objects of each band of MSI image
        :param max_vals_per_band:
        :param points_coords: list of coordinates
        """
        super().__init__(pil_msi_obj)
        self.points_coord = points_coord
        self.band_name = band_name
        self.pil_msi_obj = PointsWindow(pil_msi_obj, self.points_coord)
        self.unstretch_ims_img = None
        self.points = self.convert_pil_points_to_array(self.pil_msi_obj,self.band_name).reshape([-1,1])

    def convert_pil_points_to_array(self,pil_msi_obj,band_name):
        """
        Read values extracted from points
        :param points_coord:
        :return: array with dim [nb_bands,nb_points]
        """
        for band_idx, band in enumerate(pil_msi_obj.bands_list):
            if band == band_name:
                points = list(map(pil_msi_obj.pil_msi_img[band_idx].getpixel, self.points_coord))
        return np.array(points)

def strech_contrast(val,max_val):
    """Strech im by max value without oversaturated pixels
    max_val - int, bit depth"""
    val = np.clip(val,a_max=max_val,a_min=0.)
    val= val/max_val
    return val

def standartize(im,max_value):
    """Standartized image by grey_value"""
    im = im/max_value
    im = np.clip(im,a_max=1.0,a_min=0.0)
    return im

def conver_pil_msi_ims_to_array(pil_msi_img,width,height,nb_bands):
    """Convert PIL image cube into array image cube"""

    msi_ims = np.zeros([nb_bands,height,width])
    for band_idx, im_band in enumerate(pil_msi_img):
        im = np.array(im_band)
        msi_ims[band_idx] = im
    return msi_ims






