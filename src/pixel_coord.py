from PIL import Image
import numpy as np


class ClassCoord:
    def __init__(self,fpath,rotate_angle):
        """Reads coordinates from image mask.
        Image mask should be black on white, where black is object, white is background
        :param fpath: path to image of a mask
        :param rotate_angle of palimpsest image to align image with palimpsest

        """
        self.fpath = fpath
        self.rotate_angle = rotate_angle
        self.coords = self.read_file()

    def read_file(self):
        """
        Read coordinates of black pixels from image mask
        :return:
        coords: [[row_0,col_0],...,[row_i,col_i]]
        """
        with Image.open(self.fpath).convert('L') as im:
            if self.rotate_angle > 0:
                rotation = eval("Image.ROTATE_{}".format(self.rotate_angle))
                im = im.transpose(rotation)
            im = np.array(im)
        if not im.dtype == bool:
            if im.max()>1:
                im = im/im.max()
            im = im>0.5
        coords = np.argwhere(im==False)
        coords_x_y = list(zip(coords[:,1],coords[:,0]))
        return coords_x_y

def points_coord_in_bbox(fpath,bbox):
    """
    Read points from the mask in the range of bounding box
    :param fpath - path to image mask
    :param bbox - [top,left,bottom,right] bbox coordinates that defines the range of dataset
    """
    coords = ClassCoord(fpath, 0).coords

    coords = [(x, y) for x, y in coords if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]]
    xs,ys = map(list, zip(*coords))

    nb_coords = len(xs)
    return xs, ys, nb_coords

if __name__=="__main__":
    pass