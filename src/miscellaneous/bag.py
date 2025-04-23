import os
import numpy as np
from src.util import save_json,read_json,read_split_box_coord
from src.read_data import read_msi_image_object,read_subset_features,read_ot_mask
from src.msi_data_as_array import FullImageFromPILImageCube
from skimage import io
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import tree
import pickle



def read_ims_msi_object(main_dir,palimpsest_name,folio_name,modality,box=None):
    palimpsest_dir = os.path.join(main_dir,palimpsest_name)
    im_pil_ob = read_msi_image_object(palimpsest_dir, folio_name, modality)
    msi_im_obj = FullImageFromPILImageCube(pil_msi_obj=im_pil_ob)
    msi_im = msi_im_obj.ims_img
    if not box is None:
        bbox_fpath = os.path.join(main_dir,palimpsest_name, folio_name, "dataset_split.json")
        bbox_dict = read_json(bbox_fpath)
        bbox = read_split_box_coord(box, bbox_dict)
        msi_im = msi_im[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    msi_im = np.transpose(msi_im, axes=[1, 2, 0])
    return msi_im

def read_bg_ut_features(palimpsest_dir, folio_name,class_name_ut, class_name_bg, modality, box = None):
    features_bg,xs_bg,ys_bg = read_subset_features(palimpsest_dir, folio_name, class_name_bg, modality, box)
    features_ut,xs_ut,ys_ut = read_subset_features(palimpsest_dir, folio_name, class_name_ut, modality, box)
    return features_bg,features_ut

def read_features_mult_folio(palimpsest_dir, folio_names,class_names_ut, class_names_bg, modality, boxs = None):
    if boxs is None:
        boxs = [None]*len(folio_names)
    features_bgs = []
    features_uts = []
    for folio_name, class_name_ut, class_name_bg,box in zip(folio_names,class_names_ut,class_names_bg,boxs):
        print("box",box)
        features_bg,features_ut = read_bg_ut_features(palimpsest_dir, folio_name,class_name_ut, class_name_bg, modality, box)
        features_bgs.append(features_bg)
        features_uts.append(features_ut)
    features_bgs = np.vstack(features_bgs)
    features_uts = np.vstack(features_uts)
    return features_bgs,features_uts

def descision_tree_class(features_bg, features_ut, save_dir, nb_estimators):
    # Combine the features and create labels
    X = np.vstack((features_bg, features_ut))
    y = np.hstack((np.zeros(features_bg.shape[0]), np.ones(features_ut.shape[0])))
    model = BaggingClassifier(n_estimators=nb_estimators,n_jobs=4)
    model.fit(X, y)
    single_model = model.estimators_[0]
    feature_importances = single_model.feature_importances_
    print(feature_importances)
    model_save_path = os.path.join(save_dir, "classifier_and_params.pkl")
    with open(model_save_path, 'wb') as file:
        pickle.dump((model, model.get_params()), file)
    return model

def perform_kfold_classification(features_bg, features_ut, n_splits=5):
    # Combine the features and create labels
    X = np.vstack((features_bg, features_ut))
    y = np.hstack((np.zeros(features_bg.shape[0]), np.ones(features_ut.shape[0])))
    stratified_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    accuracies = []
    true_poss = []
    true_negs = []
    false_poss = []
    false_negs = []
    models = []
    k = 1
    for train_index, test_index in stratified_k_fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        k+=1
        print("Start model fitting for fold ",k," of ",n_splits,"...")
        model = BaggingClassifier(n_estimators=1)
        model.fit(X_train, y_train)
        models.append(model)
        y_pred = model.predict(X_test)
        tp, tn, fp, fn = calculate_tp_tn_fp_fn(y_test, y_pred)
        accuracy = calculate_weighted_accuracy(tp, tn, fp, fn)
        true_poss.append(tp)
        true_negs.append(tn)
        false_poss.append(fp)
        false_negs.append(fn)
        accuracies.append(accuracy)
    mean_accuracy = np.mean(accuracies)
    mean_true_pos = np.mean(true_poss)
    mean_true_neg = np.mean(true_negs)
    mean_false_pos = np.mean(false_poss)
    mean_false_neg = np.mean(false_negs)
    return mean_accuracy, accuracies, mean_true_pos, mean_true_neg, mean_false_pos, mean_false_neg, models

def calculate_tp_tn_fp_fn(y_true, y_pred):
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    return tp, tn, fp, fn

def calculate_weighted_accuracy(tp, tn, fp, fn):
    weighted_accuracy = 0.5*((tp / (tp + fn)) + (tn / (tn + fp)))
    return weighted_accuracy

def calculate_sensitivity(tp,fn):
    """Measures how accurate is class 1"""
    ##Sensitivity = TruePositives / (TruePositives + FalseNegatives)
    return tp/(tp+fn)

def calculate_specificity(tn,fp):
    """Measures how accurate is class 0"""
    #Specificity = TrueNegatives / (TrueNegatives + FalsePositives)
    return tn/(tn+fp)



def test_set_metrics(main_dir,palimpsest_name,folio_name,modality,models, save_dir):
    class_name_bg = "bg_lines_ot_subtracted"
    class_name_ut = "undertext_ot_subtracted"
    palimpsest_dir = os.path.join(main_dir,palimpsest_name)
    box = "test"
    features_bg,features_ut = read_bg_ut_features(palimpsest_dir,folio_name,class_name_ut,class_name_bg,modality,box)
    X = np.vstack((features_bg, features_ut))
    y = np.hstack((np.zeros(features_bg.shape[0]), np.ones(features_ut.shape[0])))
    accuracies = []
    sensitivity = []
    specificity = []
    if not type(models) is list:
        models = [models]
    for model in models:
        y_pred = model.predict(X)
        tp, tn, fp, fn = calculate_tp_tn_fp_fn(y, y_pred)
        accuracies.append(calculate_weighted_accuracy(tp, tn, fp, fn))
        sensitivity.append(calculate_sensitivity(tp,fn))
        specificity.append(calculate_specificity(tn,fp))
    d = {f"Weighted_mean_accuracy on bg {class_name_bg} and ut {class_name_ut} line test set":accuracies,
         f"Sensitivity on bg {class_name_bg} and ut {class_name_ut} line test set":sensitivity,
         f"Specificity on bg {class_name_bg} and ut {class_name_ut} line test set":specificity}
    save_json(os.path.join(save_dir, f"{folio_name}_NB_{class_name_ut}_{class_name_bg}.json"), d)

def predict_msi_image(msi_im, models,palimpsest_dir, folio_name, save_path,zeroed_ot,box="test"):
    # Reshape msi_im for prediction
    height, width, channels = msi_im.shape
    reshaped_msi_im = msi_im.reshape((-1, channels))

    # Perform prediction
    predictions = np.zeros((reshaped_msi_im.shape[0],))
    for model in models:
        predictions += model.predict(reshaped_msi_im)

    predictions = predictions/len(models)
    # Reshape predictions back to image shape
    prediction_image = predictions.reshape((height, width))
    #read ot mask and substruct it from the image
    if zeroed_ot:
        ot_im = read_ot_mask(main_dir,palimpsest_name,folio_name,box)
        ot_im[ot_im>=0.5]=1
        prediction_image[ot_im==0]=0
    prediction_image = (255 * prediction_image).astype(np.uint8)
    io.imsave(os.path.join(save_path, f"{folio_name}_NB_{class_name_ut}_{class_name_bg}.png"), prediction_image)
    return prediction_image


if __name__=="__main__":
    main_dir = r"c:\Data\PhD\palimpsest\Victor_data"
    palimpsest_name = "Paris_Coislin"
    folio_name = r"Par_coislin_393_054r"
    folio_names = [folio_name]
    modality = "M"
    nb_estimators = 1
    exp_name = f"BAG_{nb_estimators}_tree_clean_ut_train_set"
    class_name_bg = [r"bg_lines"]
    class_names_ut = [r"undertext_ot_subtracted",r"undertext_cleaned_10nn_ot_sub"]
    for class_name_ut in class_names_ut:
        class_name_ut = [class_name_ut]
        save_path = os.path.join(main_dir, palimpsest_name, folio_name, "miscellaneous", exp_name+"_"+class_name_ut[0])
        os.makedirs(save_path, exist_ok=True)
        im_msi = read_ims_msi_object(main_dir,palimpsest_name,folio_name,modality,"test")
        palimpsest_dir = os.path.join(main_dir,palimpsest_name)
        box = ["train"]
        features_bg,features_ut = read_features_mult_folio(palimpsest_dir,folio_names,class_name_ut,class_name_bg,modality,box)
        models = descision_tree_class(features_bg, features_ut, save_path, nb_estimators)
        test_set_metrics(main_dir,palimpsest_name,folio_name,modality,models, save_path)
        predict_msi_image(im_msi, models,palimpsest_dir, folio_name, save_path, True)
    #mean_accuracy, accuracies, tp, tn, fp, fn,  models = perform_kfold_classification(features_bg, features_ut)
    #predict_msi_image(im_msi, models,palimpsest_dir, folio_name, save_path, True)
    #folio_name = r"Par_coislin_393_054v"
    #im_msi = read_ims_msi_object(main_dir,palimpsest_name,folio_name,modality)
    #save_path = os.path.join(main_dir, palimpsest_name, folio_name, "miscellaneous", exp_name)
    #os.makedirs(save_path, exist_ok=True)
    zeroed_ot = True
    #predict_msi_image(im_msi, models, palimpsest_dir, folio_name,save_path,zeroed_ot)

