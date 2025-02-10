from tensorflow.python.ops.gen_clustering_ops import nearest_neighbors

from read_data import read_msi_image_object,read_x_y_coords
from msi_data_as_array import PointsfromMSI_PIL
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from skimage import io
import numpy as np
from collections import Counter
import os


def read_features(im_pil_ob,main_dir,folio_name,class_name):
    """
    Reads the features and coordinates of points from a PIL image object.

    Parameters:
    im_pil_ob: PIL.Image.Image
        The PIL image object containing the MSI image data.
    main_dir: str
        The main directory containing the image and metadata files.
    folio_name: str
        The name of the folio or subdirectory under the main directory.
    class_name: str
        The category or classification associated with the dataset.

    Returns:
    tuple
        A tuple containing:
        1. features: list
           The extracted feature points from the given PIL image.
        2. xs: list
           The x-coordinates of the points.
        3. ys: list
           The y-coordinates of the points.
    """
    xs, ys = read_x_y_coords(main_dir, folio_name, class_name, im_pil_ob, None)
    points_object = PointsfromMSI_PIL(pil_msi_obj=im_pil_ob, points_coord=list(zip(xs, ys)))
    features = points_object.points
    return features,xs,ys

def resampling_with_repeated_enn(features,labels, targeted_classes,max_iter=50,n_jobs=-1,n_neig=4):
    """
    Performs data resampling using the Repeated Edited Nearest Neighbours (RepeatedENN) technique.

    This function applies the RepeatedENN algorithm to the provided dataset to reduce noise and handle imbalance
    by removing samples inconsistent with their nearest neighbors.

    Parameters:
    features: The feature matrix containing the dataset's attributes to be resampled.
    labels: The corresponding labels for the dataset instances.
    targeted_classes: Specifies the class or classes for which the sampling strategy should be applied.

    Returns:
    - The sample indices selected after applying the RepeatedENN algorithm.
    - The resampled labels after noise removal.
    """
    enn = RepeatedEditedNearestNeighbours(sampling_strategy=targeted_classes,kind_sel="all",n_jobs=n_jobs,max_iter=max_iter,n_neighbors=n_neig)
    features_clean, labels_clean = enn.fit_resample(features, labels)
    return enn.sample_indices_, labels_clean

def repeated_enn_test():
    labels = [0,0,1,1,1,1]
    features = [[1,2],[2,4],[1.1,2.1],[8,8],[9,9],[7,7]]
    resampled_idxs,labels_clean = resampling_with_repeated_enn(features,labels,[1],max_iter=1,n_neig=2)
    print(resampled_idxs)
    print(labels_clean)
    ys = [1,2,3,4,5,6]
    xs = [1,2,3,4,5,6]
    coords_ut = [[ys[idx], xs[idx]] for idx in resampled_idxs if labels[idx] == 1]
    print(coords_ut)

def clean_labels(main_dir, folio_name,modality):
    """
    Cleans and processes the labeled dataset using oversampling and resampling techniques, and saves the updated mask.

    Parameters:
    main_dir: str
        The main directory path where the data is located.
    folio_name: str
        The name of the folio being processed.
    modality: str
        The type of imaging modality used.

    Processes:
    1. Reads and processes the multispectral image based on the provided directory, folio name, and modality.
    2. Extracts features and their associated coordinates for two distinct labels: "undertext" and "not_undertext."
    3. Merges the extracted features and coordinates of both labels into single arrays.
    4. Calculates the number of examples for "undertext" and "not_undertext" categories.
    5. Combines the labels into one array, assigning 1 for "undertext" and 0 for "not_undertext."
    6. Prints the class distribution of the original dataset.
    7. Performs data resampling using Repeated Edited Nearest Neighbors (RENN) to clean the dataset and rebalances the labels.
    8. Prints the class distribution of the cleaned dataset.
    9. Saves a newly generated mask of the cleaned labels into an output file named "undertext_renn_black.png" in the specified directory.
    """
    im_pil_ob = read_msi_image_object(main_dir, folio_name, modality)
    features_ut,xs_ut,ys_ut = read_features(im_pil_ob,main_dir, folio_name,"undertext")
    features_nonut,xs_nonut,ys_nonut = read_features(im_pil_ob,main_dir, folio_name,"not_undertext")
    xs = np.concatenate((xs_ut,xs_nonut))
    ys = np.concatenate((ys_ut,ys_nonut))
    nb_ut = len(ys_ut)
    nb_nonut = len(ys_nonut)
    features = np.concatenate((features_ut,features_nonut),axis=0)
    labels = np.concatenate(([1]*nb_ut,[0]*nb_nonut),axis=0)
    print("Original dataset,", Counter(labels))
    cleaned_idxs, labels_cleaned = resampling_with_repeated_enn(features,labels,[1])
    print("Cleaned dataset,", Counter(labels_cleaned))
    save_new_mask(cleaned_idxs,xs,ys,labels,im_pil_ob.width,im_pil_ob.height,main_dir,folio_name,folio_name+"-undertext_renn_black.png")

def save_new_mask(idxs,xs,ys,labels,width,height,main_dir,folio_name,fname):
    coords_ut = [[ys[idx],xs[idx]] for idx in idxs if labels[idx]==1]
    coords_ut = np.array(coords_ut)
    im = np.ones((height,width))
    im[coords_ut[:,0],coords_ut[:,1]] = 0
    im = (im*255).astype(np.uint8)
    io.imsave(os.path.join(main_dir,folio_name,"mask",fname),im)




if __name__ == "__main__":
    repeated_enn_test()
    main_data_dir = "/projects/palimpsests/Verona_msXL" #"D:\Verona_msXL"#
    folio_name = r"msXL_335v_b"
    modality = "M"
    folio_names = ["msXL_335v_b", r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
                   "msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b",
                   "msXL_334v_b", "msXL_344r_b", "msXL_344v_b", r"msXL_315r_b"]
    for folio_name in folio_names:
        clean_labels(main_data_dir, folio_name,modality)
