from pyexpat import features

from pil_image_cube import ImageCubePILobject
from msi_data_as_array import PointsfromMSI_PIL,PointsfromRatio,PointsfromBand
from pixel_coord import points_coord_in_bbox
from util import read_json,read_band_list,read_split_box_coord
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
plt.switch_backend('TkAgg')
import os
from yellowbrick.features import RadViz
import numpy as np
osp = os.path.join

def read_msi_and_coords(main_dir,folio_name,fname_mask,modality,box=None):
    """
    Reads subset features from a given directory for a specified folio and class name.

    Parameters:
    main_dir (str): The main directory path containing the necessary data files.
    folio_name (str): The name of the folio to read.
    class_name (str): The class name used to format the mask filename.
    box (list, optional): A list specifying the bounding box coordinates [x1, y1, x2, y2]. Defaults to None.

    Returns:
    list: A list of feature points extracted from the specified input data.
    """
    band_list_path = osp(main_dir, r"band_list.txt")
    bands = read_band_list(band_list_path,modality)

    im_pil_ob = ImageCubePILobject(main_dir,folio_name,bands,0)
    if not box is None:
        bbox_fpath = osp(main_dir, folio_name, "dataset_split.json")
        bbox_dict = read_json(bbox_fpath)
        bbox = read_split_box_coord(box, bbox_dict)
    else:
        bbox = [0,0,im_pil_ob.width-1,im_pil_ob.height-1]
    fpath_image_mask = osp(main_dir,folio_name,"mask",folio_name + r"-{}_black.png".format(fname_mask))
    xs, ys, nb_coords = points_coord_in_bbox(fpath_image_mask, bbox)
    return im_pil_ob,xs,ys

def read_subset_features(main_dir,folio_name,class_name,modality,box=None):
    im_pil_ob, xs, ys = read_msi_and_coords(main_dir,folio_name,class_name,modality,box)
    points_object = PointsfromMSI_PIL(pil_msi_obj=im_pil_ob, points_coord= list(zip(xs,ys)))
    features = points_object.points
    return features


def read_subset_features_ratio(main_dir,folio_name,class_name,modality,box=None):
    im_pil_ob, xs, ys = read_msi_and_coords(main_dir, folio_name, class_name, modality, box)
    points_object = PointsfromRatio(pil_msi_obj=im_pil_ob, points_coord=list(zip(xs, ys)))
    features_ratio_W420B47_W385UVB = points_object.points_ratio_W420B47_W385UVB
    features_ratio_W365UVP_W385UVB = points_object.points_ratio_W365UVP_W385UVB
    features = np.concatenate([features_ratio_W420B47_W385UVB, features_ratio_W365UVP_W385UVB], 1)
    return features

def read_subset_features_from_band(main_dir,folio_name,class_name,band_name,box=None):
    im_pil_ob, xs, ys = read_msi_and_coords(main_dir, folio_name, class_name, modality, box)
    points_object = PointsfromBand(pil_msi_obj=im_pil_ob, points_coord=list(zip(xs, ys)),band_name=band_name)
    features = points_object.points
    return features

def viz_RadViz(features,labels,classes,result_path):
    """
    Generates and displays a RadViz visualization for the given feature set and labels.

    Args:
        features (array-like): The feature set for the data points.
        labels (array-like): The class labels corresponding to the data points.
        classes (list of str): The class names for the classification problem.
        result_path (str): Path to save the resulting visualization image.

    Functionality:
        Creates a RadViz visualization using the provided features, labels, and class names.
        Sets specific visualization parameters for alpha transparency, light background, and y-axis as "C02".
        Fits and transforms the data using the RadViz visualizer.
        Displays the visualization and saves it to the defined file path if provided.
    """
    vizualizer = RadViz(classes=classes,alpha=0.5,x="light", y="C02")
    vizualizer.fit_transform(features,labels)

    vizualizer.show(outpath=result_path)


def visualize_tsne(features,labels,classes,result_path):
    """Generates and displays a t-SNE visualization for the given feature set and labels."""
    tsne = TSNE(n_components=2, random_state=42)
    transformed_features = tsne.fit_transform(features)

    # Plot the results
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca()
    scatter = plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=labels, cmap='viridis')
    handles, _ = scatter.legend_elements()
    # Add legend
    legend1 = plt.legend(handles, classes, title="Classes")
    ax.add_artist(legend1)

    # Add titles and labels
    plt.title("t-SNE visualization")
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.show()
    fig.savefig(result_path)

def read_features_ut_vs_non_ut(main_dir,folio_name,normalized,modality, fname_ut,fname_non_ut):
    """

        Reads and processes features for undertext (UT) vs non-undertext (non-UT) samples.

        Args:
            main_dir (str): The main directory path where the feature files are located.
            folio_name (str): The specific folio name to read the features from.
            normalized (bool): A flag indicating whether to normalize the features.

        Returns:
            tuple: A tuple containing:
                - features (np.ndarray): The concatenated features for UT and non-UT samples.
                - labels (np.ndarray): The corresponding labels for the features (0 for non-UT, 1 for UT).
                - classes (list): The list of class names, ["not_undertext", "undertext"].

        Notes:
            This function reads two sets of features (UT and non-UT), randomly samples 50% from each,
            concatenates them, and assigns labels. The features can optionally be normalized before returning.
    """
    ut_features = read_subset_features(main_dir,folio_name,fname_ut,modality,"val")
    print("ut feature extraction finished")
    nonut_features = read_subset_features(main_dir,folio_name,fname_non_ut,modality,"val")
    print("non ut feature extraction finished")
    nb_samples = min(3000,len(ut_features))
    np.random.seed(5)
    rand_ut_idx = np.random.choice(np.arange(len(ut_features)),size=nb_samples//2)
    np.random.seed(5)
    rand_nonut_idx = np.random.choice(np.arange(len(nonut_features)),size=nb_samples//2)
    features = np.concatenate([ut_features[rand_ut_idx],nonut_features[rand_nonut_idx]],0)
    labels = np.array([0]*len(rand_ut_idx)+[1]*len(rand_nonut_idx))
    if normalized:
        features = normilize(features)
    classes = ["not_undertext", "undertext"]
    return features,labels,classes

def read_features_ut_ot_parc_varnish(main_dir,folio_name,normalized,modality):
    """
    Reads and processes feature subsets from different categories: undertext (without varnish and with varnish), parchment (with and without varnish), and overtext from the given directory. It returns concatenated features, corresponding labels, and class names.

    Args:
        main_dir (str): The main directory containing feature subsets.
        folio_name (str): The name identifier of the folio to read features from.
        normalized (bool): Flag to indicate whether to normalize the features.

    Returns:
        tuple: A tuple containing:
            - features (numpy.ndarray): Concatenated features from all categories.
            - labels (numpy.ndarray): Labels associated with each feature indicating its class.
            - classes (list): List of class names corresponding to the labels.
    """
    max_nb_samples = 500
    ut_features = read_subset_features(main_dir, folio_name, "undertext_novarnish", modality, "val")
    np.random.seed(5)
    rand_ut_idx = np.random.choice(np.arange(len(ut_features)), size=min(max_nb_samples,len(ut_features)))
    ut_features = ut_features[rand_ut_idx]
    utvar_features = read_subset_features(main_dir, folio_name, "undertext_varnish",modality, "val")
    np.random.seed(5)
    rand_utvar_idx = np.random.choice(np.arange(len(utvar_features)), size=min(max_nb_samples,len(utvar_features)))
    utvar_features = utvar_features[rand_utvar_idx]
    parch_features = read_subset_features(main_dir, folio_name, "parchment_novarnish",modality,None)
    parch_features = parch_features[:min(max_nb_samples,len(parch_features))]
    parch_varnish_features = read_subset_features(main_dir, folio_name, "parchment_varnish",modality,None)
    parch_varnish_features = parch_varnish_features[:min(max_nb_samples,len(parch_varnish_features))]
    ot_features = read_subset_features(main_dir, folio_name, "overtext", modality, None)
    ot_features = ot_features[:min(max_nb_samples,len(ot_features))]
    features = np.concatenate([ut_features,ot_features, parch_features,parch_varnish_features,utvar_features], 0)
    if normalized:
        features = normilize(features)
    labels = np.array([0] * len(ut_features) + [1] * len(ot_features)+[2]*len(parch_features)+[3]*len(parch_varnish_features)+[4]*len(utvar_features))
    classes = ["undertext","overtext","parchment","parchment_varnish","undertext_varnish"]
    return features,labels,classes

def read_features_varnish_ratio(main_dir,folio_name):
    """
    Reads and processes feature subsets from different categories: undertext (without varnish and with varnish), parchment (with and without varnish), and overtext from the given directory. It returns concatenated features, corresponding labels, and class names.

    Args:
        main_dir (str): The main directory containing feature subsets.
        folio_name (str): The name identifier of the folio to read features from.
        normalized (bool): Flag to indicate whether to normalize the features.

    Returns:
        tuple: A tuple containing:
            - features (numpy.ndarray): Concatenated features from all categories.
            - labels (numpy.ndarray): Labels associated with each feature indicating its class.
            - classes (list): List of class names corresponding to the labels.
    """
    utvar_features = read_subset_features_ratio(main_dir, folio_name, "undertext_varnish",modality, None)
    parch_varnish_features = read_subset_features_ratio(main_dir, folio_name, "parchment_varnish",modality,None)
    features = np.concatenate([utvar_features,parch_varnish_features], 0)
    labels = np.array([0] * len(utvar_features) + [1] * len(parch_varnish_features))
    classes = ["undertext_varnish","parchment_varnish"]
    return features,labels,classes

def read_features_varnish_3bands(main_dir,folio_name,band_list):
    """
    Reads and processes feature subsets from different categories: undertext (without varnish and with varnish), parchment (with and without varnish), and overtext from the given directory. It returns concatenated features, corresponding labels, and class names.

    Args:
        main_dir (str): The main directory containing feature subsets.
        folio_name (str): The name identifier of the folio to read features from.
        normalized (bool): Flag to indicate whether to normalize the features.

    Returns:
        tuple: A tuple containing:
            - features (numpy.ndarray): Concatenated features from all categories.
            - labels (numpy.ndarray): Labels associated with each feature indicating its class.
            - classes (list): List of class names corresponding to the labels.
    """

    features = []
    for band_name in band_list:
        utvar_features = read_subset_features_from_band(main_dir, folio_name, "undertext_varnish", band_name,None,)
        parch_varnish_features = read_subset_features_from_band(main_dir, folio_name, "parchment_varnish",band_name,None,)
        features_band = np.concatenate([utvar_features, parch_varnish_features], 0)
        features.append(features_band)

    features = np.concatenate(features, 1)
    labels = np.array([0] * len(utvar_features) + [1] * len(parch_varnish_features))
    classes = ["undertext_varnish","parchment_varnish"]
    return features,labels,classes

def visualize_ratio_on_varnish(features,labels,classes):
    # Generate scatter plot with the required specifications
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels,
                          cmap='viridis')
    handles, _ = scatter.legend_elements()
    plt.legend(handles, classes, title="Classes")

    # Add axis labels, title, and grid
    plt.xlabel('W420B47_W385UVB')
    plt.ylabel('W365UVP_W385UVB')
    plt.title('Scatter plot of features')
    plt.grid(True)
    # Display the plot
    plt.show()


def visualize_3d_varnish_features(features, labels, classes, band_list):
    """
    Visualizes the result of read_features_varnish_3bands on a 3D scatter plot.

    Parameters:
    features (numpy.ndarray): Concatenated features from different categories.
    labels (numpy.ndarray): Labels corresponding to each feature indicating its class.
    classes (list): List of class names corresponding to the labels.
    """


    # Ensure features shape matches the expected dimensions
    assert features.shape[1] == len(band_list), "Features array does not match the expected number of bands."

    from mpl_toolkits.mplot3d import Axes3D

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with features array along axis 1
    scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap='viridis')

    # Add labels to the axes
    ax.set_xlabel(band_list[0])
    ax.set_ylabel(band_list[1])
    ax.set_zlabel(band_list[2])

    # Add title and legend
    ax.set_title('3D Scatter Plot of Varnish Features')
    handles, _ = scatter.legend_elements()
    legend = ax.legend(handles, classes, title="Classes")

    plt.show()
# Example usage:
# features, labels, classes = read_features_varnish_3bands(main_dir, folio_name)
# visualize_3d_varnish_features(features, labels, classes)

def analyze_ut_parch_varnish_3band(main_dir,folio_name):

    band_list = ["W420B47_42_F","W385UVB_21_F","W365UVP_27_F"]
    features, labels, classes = read_features_varnish_3bands(main_dir, folio_name, band_list)
    visualize_3d_varnish_features(features, labels, classes, band_list)

def analyze_ut_parch_ratio(main_dir,folio_name):

    features,labels,classes = read_features_varnish_ratio(main_dir,folio_name)
    visualize_ratio_on_varnish(features,labels,classes)

def analyze_ut_parch_ratio_log(main_dir,folio_name):

    features,labels,classes = read_features_varnish_ratio(main_dir,folio_name)
    features = np.log(features)
    visualize_ratio_on_varnish(features,labels,classes)

def analyze_labels_ut_vs_nonut(main_dir,folio_name,modality,normalized,fname_ut,fname_non_ut):
    """
    Analyzes and visualizes labeled data, comparing 'UT' vs 'non-UT' across a specific modality.

    Parameters:
    main_dir (str): The main directory containing the data.
    folio_name (str): The specific folio name to be analyzed.
    modality (str): The modality type for the analysis (e.g., image type).
    normalized (bool): Indicator whether the features should be normalized.

    Returns:
    None
    """
    features,labels,classes = read_features_ut_vs_non_ut(main_dir,folio_name,normalized,modality,fname_ut,fname_non_ut)
    dir_path = osp(main_dir,folio_name,"miscellaneous")
    fpath = "_{}{}_{}_{}.png".format(modality,"_norm" if normalized else "",fname_ut,fname_non_ut)
    #viz_RadViz(features,labels,classes,osp(dir_path,"Radvis"+fpath)
    visualize_tsne(features,labels,classes, osp(dir_path,"tSNE"+fpath))

def analyze_ut_vs_ot_parch_varnish(main_dir,folio_name,modality,normalized,features_ratio=False):
    """

    Analyze and visualize feature data using RadViz for given parameters.

    Args:
        main_dir (str): The main directory path where the data is stored.
        folio_name (str): The name of the folio to be analyzed.
        modality (str): The modality to be used in the analysis (e.g., 'MRI', 'CT').
        normalized (bool): A flag indicating whether the data should be normalized.

    Returns:
        None
    """


    features,labels,classes = read_features_ut_ot_parc_varnish(main_dir,folio_name,normalized,modality,features_ratio=features_ratio)
    dir_path = osp(main_dir, folio_name, "miscellaneous")
    fpath = "_{}_utnovar_utvar_ot_parch{}_{}.png".format(modality, "_norm" if normalized else "",
                                                           "ration" if features_ratio else "")
    #viz_RadViz(features,labels,classes,osp(dir_path,"Radviz"+fpath))
    visualize_tsne(features,labels,classes,osp(dir_path,"tSNE"+fpath))

def normilize(features):
    """
     Normalizes the given feature matrix.

     This function takes a matrix of features and normalizes each feature vector
     (row) by its Euclidean norm. This normalization can be useful in machine
     learning tasks where the scale of features may impact the algorithm's
     performance.

     Args:
         features (numpy.ndarray): A 2D array where each row represents a feature
                                   vector to be normalized.

     Returns:
         numpy.ndarray: A 2D array with the same shape as the input, where each
                        feature vector (row) is normalized by its Euclidean norm.
    """
    norms = np.sqrt(np.sum(features**2,axis=1))
    features = features/np.repeat(norms[:,np.newaxis],axis=1,repeats=features.shape[1])
    return features



if __name__=="__main__":
    main_dir = r"c:\Data\PhD\palimpsest\Victor_data\Paris_Coislin"
    folio_name = r"Par_coislin_393_054r"
    modality = "M"
    normalized = False
    #analyze_ut_parch_ratio_log(main_dir,folio_name)
    #analyze_ut_parch_varnish_3band(main_dir,folio_name)
    #analyze_ut_parch_ratio(main_dir,folio_name)
    #analyze_ut_vs_ot_parch_varnish(main_dir, folio_name, modality, normalized,features_ratio)
    fname_ut = "undertext"
    fname_non_ut = "not_undertext"
    analyze_labels_ut_vs_nonut(main_dir,folio_name,modality,normalized,fname_ut,fname_non_ut)






