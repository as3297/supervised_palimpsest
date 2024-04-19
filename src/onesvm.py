import numpy as np
import cv2
from util import read_json,read_band_list
import os
from read_files import ReadImageCube
from sklearn.svm import OneClassSVM
from matplotlib import pyplot as plt


def read_features(img,coord):
    """
    :param img: class
    :return:
    """
    features = []
    for box_idx, box_coord in enumerate(coord):
        im = img.read_msi_image(box_coord["box_"+str(box_idx)])
        features.extend(img_to_samples(im))
    samples_idx = np.arange(0,len(features))
    np.random.shuffle(samples_idx)
    samples_idx = samples_idx[:1000].tolist()
    features = np.array(features)[samples_idx,:]
    return features

def img_to_samples(im):
    im = im.reshape([im.shape[0], -1])
    features = np.transpose(im)
    return features


def oneClassSVM(img,features):
    clf = OneClassSVM(gamma='auto', nu=0.01, degree=2).fit(features)
    coord = [1000,0,4000,5000]
    im = img.read_msi_image(coord)
    out = clf.predict(img_to_samples(im))
    out = out.reshape((im.shape[1],im.shape[2]))
    out[out<0]=0
    return im[14],out

def plot_predictions(im,prediction):
    fig,ax = plt.subplots(1,2)
    ax[0].set_title("Input")
    ax[1].set_title("Prediction")
    ax[0].imshow(im,cmap="gray")
    ax[1].imshow(prediction,cmap="gray")

    plt.figure()
    plt.imshow(prediction)
    plt.show()


if __name__ == '__main__':
    image_dir = r"C:\Data\PhD\palimpsest\Victor_data"
    json_file = os.path.join(image_dir,"coord_forsvm.json")
    d = read_json(json_file)
    bg_coords = d['background']
    bands = read_band_list(os.path.join(image_dir,"band_list.txt"))
    bands = [band for band in bands if "M"]

    img = ReadImageCube(image_dir, [], 270)
    features = read_features(img, bg_coords)
    im, out = oneClassSVM(img,features)
    plot_predictions(im, out)