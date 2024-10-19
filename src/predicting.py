import copy
import numpy as np
from msi_data_as_array import PointsfromMSI_PIL
from util import read_band_list, read_json, generate_coord_inside_bbox, read_split_box_coord
from pil_image_cube import ImageCubePILobject
import tensorflow as tf
import os
from PIL import Image
from skimage import io


class LabeledIm:
  def __init__(self, fpath, rotate_angle):
    """
    :param image_dir: directory with tif image of palimpsest
    :param band_list: list of bands
    :param coord: (left, upper, right, lower) tuple of bounding box coordinates
    """
    self.fpath = fpath
    self.rotate_angle = rotate_angle
    self.im = self.read_file()

  def read_file(self):
    """
    Read image mask
    :return:
    coords: [[row_0,col_0],...,[row_i,col_i]]
    """
    with Image.open(self.fpath) as im:
      im_mode = im.mode
      if self.rotate_angle > 0:
        rotation = eval("Image.ROTATE_{}".format(self.rotate_angle))
        im = im.transpose(rotation)
      im = np.array(im)
    if im_mode == "RGBA":
      im = im[:, :, 3]
      im = np.amax(im) - im
    im = im / np.amax(im)
    return im
def load_data_for_visualization(split,label_mask,manuscriptname,folioname,full_page=False):
  osp = os.path.join
  main_dir = r"C:\Data\PhD\palimpsest\Victor_data"
  manus_dir_path = osp(main_dir,manuscriptname)
  image_dir_path = osp(manus_dir_path,folioname)
  band_list_path = osp(manus_dir_path,"band_list.txt")
  bands = read_band_list(band_list_path,"M")
  max_val_path = osp(main_dir,"bands_max_val.json")

  bbox_fpath = osp(manus_dir_path,folioname,"dataset_split.json")

  im_msi_pil_ob = ImageCubePILobject(manus_dir_path, folioname, bands, 0)
  if full_page:
    bbox = [0,0,5326,7100]
  else:
    bbox_dict = read_json(bbox_fpath)
    bbox = read_split_box_coord(split, bbox_dict)
  width = bbox[2]-bbox[0]
  height = bbox[3]-bbox[1]
  points_coord = generate_coord_inside_bbox(bbox[0],bbox[1],width,height)
  points_ob = PointsfromMSI_PIL(im_msi_pil_ob, points_coord)
  features = points_ob.points

  if label_mask:
    fpath_ut = osp(image_dir_path,"mask",folioname+r"-undertext_black.png")
    mask_ut = LabeledIm(fpath_ut, 0).im
    mask_ut = mask_ut[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    fpath_nonut = osp(image_dir_path,"mask",folioname+r"-not_undertext_black.png")
    mask_nonut = LabeledIm(fpath_nonut, 0).im
    mask_nonut = mask_nonut[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return features, width, height, mask_ut, mask_nonut
  else:
    return features, width, height

def overlay_mask(im,mask,color):
  color = np.array(color)
  new_im = copy.deepcopy(im)
  inv_mask = 1-mask #usually the mask is inverted
  #dst = src1 * alpha + src2 * beta + gamma
  alpha = 0.6
  beta = 1 - alpha
  inv_mask = (np.repeat(inv_mask[:,:,np.newaxis],3,axis=2)*255).astype(np.uint8)
  inv_mask[mask==0] = color*beta
  new_im[mask == 0] = im[mask == 0] * alpha+inv_mask[mask==0]
  return new_im

def visualizing_testing_results(saved_model_path,split,manuscriptname,folioname,label_mask=True,full_page=False):
  #imported = tf.saved_model.load(saved_model_path)
  #imported = imported.signatures["serving_default"]
  imported = tf.keras.models.load_model(saved_model_path)
  print("Loaded model from path: {}".format(saved_model_path))
  if label_mask:
    features, width,height,mask_ut,mask_nonut = load_data_for_visualization(split,label_mask,manuscriptname,folioname,full_page)

  else:
    features, width, height = load_data_for_visualization(split, label_mask,manuscriptname,folioname,full_page)
  print("Loaded data from folio: {}".format(folioname))
  batch_size = 256
  nb_samples = len(features)
  res = nb_samples%batch_size
  pred_im = np.zeros((nb_samples,))
  for idx in range(0,nb_samples-res, batch_size):
    batch = features[idx:idx + batch_size, :]
    batch = tf.constant(batch,dtype=tf.float32)
    output = imported(batch)#["output_0"]
    output = tf.sigmoid(output)
    output = output.numpy()
    pred_im[idx:idx + batch_size] = np.squeeze(output)
  pred_im = np.reshape(pred_im,[height,width,1])
  pred_im = (np.repeat(pred_im,3,axis=2)*255).astype(np.uint8)
  io.imsave(os.path.join(os.path.split(saved_model_path)[0],folioname+"_pred_im.png"),pred_im)

  if label_mask:
    pred_im_ut_mask = overlay_mask(pred_im,mask_ut,[0,255,0])
    pred_im_ut_nonut_mask = overlay_mask(pred_im_ut_mask,mask_nonut,[255,0,0])
    io.imsave(os.path.join(os.path.split(saved_model_path)[0],folioname+"_pred_im_ut_mask.png"),pred_im_ut_mask)
    io.imsave(os.path.join(os.path.split(saved_model_path)[0], folioname+"_pred_im_ut_nonut_mask.png"), pred_im_ut_nonut_mask)


if __name__=="__main__":
  saved_model_path = r"C:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training\20240926-113459\model.keras"
  manuscriptname = r"Verona_XL_(38)"
  folionames = [r"msXL_315r_b"]
  full_page = False
  for folioname in folionames:
    visualizing_testing_results(saved_model_path,"val",manuscriptname,folioname,label_mask=True,full_page=full_page)