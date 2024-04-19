from PIL import Image
import numpy as np
import os
import xml.etree.ElementTree as ET

class ReadImageCubePILobject():
    def __init__(self,image_dir,band_list,rotate_angle,max_val_per_band):
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
        self.band_list = sorted(band_list)
        self.rotate_angle = rotate_angle
        self.max_values = max_val_per_band#self.max_of_band_im(25)
        self.stretch_contrast = True

    #open file as multipage tif
    #close file
    def thumbnail(self,band_idx,scale_ratio):
        """Opens image thumbnail"""
        path = os.path.join(self.image_dir,self.band_list[band_idx]+".tif")
        im = self.read_image(path,False,scale_ratio,[0,0,0,0],self.max_values[band_idx])
        return im

    def max_of_band_im(self,scale_ratio):
        """Reads image of multiple bands"""
        self.stretch_contrast = False
        max_vals = np.zeros((len(self.band_list)))
        for idx, band_name in enumerate(self.band_list):
            path = os.path.join(self.image_dir, self.band_list[idx] + ".tif")
            im = self.read_image(path, False, scale_ratio, [0, 0, 0, 0],1)
            max_vals[idx] = find_max(im,np.amax(im))
        return max_vals

    def read_image_object(self,path,max_val):
        with Image.open(path) as im:
            if self.rotate_angle>0:
                rotation = eval("Image.ROTATE_{}".format(self.rotate_angle))
                im = im.transpose(rotation)
            if self.stretch_contrast:
                im = self.strech_contrast_fun(im,max_val)
        return im

    def strech_contrast_fun(self,im,max_val):
        """Strech im by max value without oversaturated pixels
        max_val - int, bit depth"""
        im = np.clip(im,a_max=max_val,a_min=0.)
        im = im/max_val
        return im

    def read_msi_image_object(self):
        """Reads image of multiple bands
                coord - [x1,y1,x2,y2]"""
        ims = np.zeros(((len(self.band_list),) + size))
        for idx, band_name in enumerate(self.band_list):
            fpath = os.path.join(self.image_dir, band_name + ".tif")
            im = self.read_image(fpath, True, 1, coord, self.max_values[idx])
            ims[idx] = im
        return ims

    def read_image(self,path,bbox,scale_ratio,coord,max_val):
        """Reads image of one band
        Args:
            coords - [left, upper, right, lower]
            """

        with Image.open(path) as im:
            if self.rotate_angle>0:
                rotation = eval("Image.ROTATE_{}".format(self.rotate_angle))
                im = im.transpose(rotation)
            if len(bbox):
                "coords - left, upper, right, lower"
                im = im.crop(coord)
            if scale_ratio>1:
                width, height = im.size
                new_size = (width//scale_ratio,height//scale_ratio)
                im = im.resize(new_size)
            im = np.array(im)
            if self.stretch_contrast:
                im = self.strech_contrast_fun(im,max_val)

        return im

    def read_msi_fragment(self,coord):
        """Reads image of multiple bands
        coord - [x1,y1,x2,y2]"""
        size = (coord[3]-coord[1],coord[2]-coord[0])
        ims = np.zeros(((len(self.band_list),)+size))
        for idx, band_name in enumerate(self.band_list):
            fpath = os.path.join(self.image_dir,band_name+".tif")
            im = self.read_image(fpath,True,1,coord,self.max_values[idx])
            ims[idx]= im
        return ims



def find_max(im,max_val):
    """Find max value without oversaturated pixels
    max_val - scalar, bit depth"""
    if max_val == 1:
        bin_width = 1 / 256.0
    elif max_val == 256:
        bin_width = 1
    else:
        bin_width = 10
    bins = np.arange(0, max_val, bin_width)
    hist, bins = np.histogram(im, bins=bins)
    hist = hist/np.sum(hist)
    for idx in reversed(range(len(hist))):
        if hist[idx] < 0.001 and hist[idx - 1] == 0:
        # print("Hist now {}, hist before {}".format(hist[-idx],hist[-idx-1]))
            max_val = bins[idx - 1]
        else:
            break
    return max_val

class BoxfromROI():
    def __init__(self,roipath):
        self.fpath = roipath

    def read_bg_bbox_corners_coord(self):
        """Read coordinates of bounding boxes around background"""
        bs_data = ET.parse(self.fpath)
        root = bs_data.getroot()
        region = root.find('Region')
        geometry = region.find("GeometryDef")
        bboxs = []
        for polygon in geometry.iter("Polygon"):
            exterior = polygon.find("Exterior")
            linear_ring = exterior.find("LinearRing")
            bbox = linear_ring.find('Coordinates')
            bbox = bbox.text.split()
            cols = [int(float(bbox[i])) for i in range(len(bbox)) if i==0 or i%2==0]
            rows = [int(float(bbox[i])) for i in range(len(bbox)) if i%2==1]
            min_row = np.min(rows)
            min_col = np.min(cols)
            max_row = np.max(rows)
            max_col = np.max(cols)
            bboxs.append(np.array([[min_row,max_row],[min_col,max_col]]))
        return bboxs

def read_stretch_contr_image(datapath,folioname,bandname):
    im = io.imread(os.path.join(datapath,folioname,folioname+"+"+bandname+".tif"),as_gray=True)

    im = strech_contrast(im,max_val(im))
    return im

def max_val(im):
    if np.amax(im) > 255:
        max_val = 65535
        #im = im / 65535.0
        #im = im*255
    elif np.amax(im) > 1 and np.amax(im) < 256:
        max_val = 256
    elif np.amax(im)<=1:
        max_val = 1
    else:
        raise IOError("Expected format is uint8 or uint16")
    return max_val

def read_bg_features_greek960(datapath,bandnames,folio_idx):
    folioname = r"0086_000{}".format(folio_idx)
    bboxs_coord = read_bg_bbox_corners_coord(os.path.join(datapath, folioname, "background_roi.xml"))
    features = []

    for bbox_idx, bbox in enumerate(bboxs_coord):
        features_box = read_fragment_msi_greek960(os.path.join(datapath, folioname, folioname),row_coord=bbox[0],
                                 col_coord=bbox[1],
                                 bandnames=bandnames,scale=None)

        plt.figure("Bg")
        plt.imshow(features_box[:,:,0]/500,cmap="gray", vmin=0.0,vmax=1.0)
        features.extend(features_box.reshape(-1,len(bandnames)))

    return features
def novelty_detection_with_RXD_greek960():
    folio_idx= "084"
    folioname = r"0086_000{}".format(folio_idx)
    fname_ot = r"0086_000084+MB625Rd_007_F_thresh.png"
    datapath = r"C:\Data\PhD\bss_autoreg_palimpsest\datasets\Greek_960"#r"C:\Data\PhD\palimpsest\Greek_960"#
    ot_path = os.path.join(datapath,folioname,fname_ot)
    bandmode = "all_bands"
    if bandmode=="all_bands":
        bandnames = read_bandnames_greek960(os.path.join(datapath, "0086_000084", "band_names.txt"))
    elif bandmode=="ld_msi_bands":
        bandnames = ["MB365UV_001_F","MB625Rd_007_F","MB455RB_002_F"]
    features = read_bg_features_greek960(datapath, bandnames, folio_idx,"background_roi.xml")
    row_coord = [2300, 2300 + 192]
    col_coord = [1800, 2914]
    scale = None#64 / 192
    pal = read_fragment_msi_greek960(os.path.join(datapath, folioname, folioname),row_coord,col_coord,bandnames,scale,rotate_angle=-90)
    ot = read_fragment(ot_path,row_coord,col_coord,0,scale)
    ot = ot/np.amax(ot)
    plt.figure("Palimpsest")
    plt.imshow(pal[:,:,0]/500,cmap="gray", vmin=0,vmax=1.0)
    plt.figure("Overtext")
    plt.imshow(ot,cmap="gray", vmin=0,vmax=1.0)
    dist_org = calc_Mahalanobis_dist(pal,features)
    print("Dist. min", np.min(dist_org))
    plt.figure("Dist")
    plt.imshow(dist_org, cmap="gray")

    thresh = threshold_otsu(dist_org[ot==1])
    dist = dist_org > thresh
    plt.figure("Thresh dist after otsu")
    plt.imshow(dist, cmap="gray")
    dist[ot<1]=0
    plt.figure("Dist ot masked")
    plt.imshow(dist, cmap="gray")

    save_path = r"C:\Data\PhD\bss_autoreg_palimpsest\related_exp\RX\greek960"
    io.imsave(os.path.join(save_path,"0086_000"+folio_idx+"rxd_{}_{}_{}.png".format(row_coord,col_coord,bandmode)),dist_org)
    io.imsave(
        os.path.join(save_path, "0086_000" + folio_idx + "rxd_otsu_masked_{}_{}_{}.png".format(row_coord, col_coord, bandmode)),
        dist)
    #np.savez(os.path.join(datapath,"0086_000"+folio_idx,"rxd_{}_{}.npy",dist)
    plt.show()


