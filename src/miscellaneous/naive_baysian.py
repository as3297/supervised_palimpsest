import os
import numpy as np
from src.util import save_json
from src.read_data import read_msi_image_object,read_subset_features
from src.msi_data_as_array import FullImageFromPILImageCube
from skimage import io
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB



def read_ims_msi_object(main_dir,palimpsest_name,folio_name,modality):
    palimpsest_dir = os.path.join(main_dir,palimpsest_name)
    im_pil_ob = read_msi_image_object(palimpsest_dir, folio_name, modality)
    msi_im_obj = FullImageFromPILImageCube(pil_msi_obj=im_pil_ob)
    msi_im = msi_im_obj.ims_img
    msi_im = np.transpose(msi_im, axes=[1, 2, 0])
    return msi_im

def read_bg_ut_features(palimpsest_dir, folio_name,class_name_ut, class_name_bg, modality,box):
    features_bg,xs_bg,ys_bg = read_subset_features(palimpsest_dir, folio_name, class_name_bg, modality, box)
    features_ut,xs_ut,ys_ut = read_subset_features(palimpsest_dir, folio_name, class_name_ut, modality, box)
    return features_bg,features_ut

def perform_kfold_classification(features_bg, features_ut, n_splits=5):
    # Combine the features and create labels
    X = np.vstack((features_bg, features_ut))
    y = np.hstack((np.zeros(features_bg.shape[0]), np.ones(features_ut.shape[0])))
    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    true_poss = []
    true_negs = []
    false_poss = []
    false_negs = []
    models = []
    for train_index, test_index in stratified_k_fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = GaussianNB()
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


def NB_train(features_bg, features_ut):
    # Combine the features and create labels
    X = np.vstack((features_bg, features_ut))
    y = np.hstack((np.zeros(features_bg.shape[0]), np.ones(features_ut.shape[0])))
    model = GaussianNB()
    model.fit(X, y)
    y_pred = model.predict(X)
    tp, tn, fp, fn = calculate_tp_tn_fp_fn(y, y_pred)
    weighted_accuracy = calculate_weighted_accuracy(tp, tn, fp, fn)
    sensitivity = calculate_sensitivity(tp,fn)
    specificity = calculate_specificity(tn,fp)
    return model,weighted_accuracy, sensitivity, specificity

def NB_test(main_dir,palimpsest_name,folio_name,model):
    class_name_bg = "bg_lines_ot_subtracted"
    class_name_ut = "undertext_ot_subtracted"
    palimpsest_dir = os.path.join(main_dir, palimpsest_name)
    features_bg, features_ut = read_bg_ut_features(palimpsest_dir, folio_name, class_name_ut, class_name_bg, modality,"test")
    X = np.vstack((features_bg, features_ut))
    y = np.hstack((np.zeros(features_bg.shape[0]), np.ones(features_ut.shape[0])))
    y_pred = model.predict(X)
    tp, tn, fp, fn = calculate_tp_tn_fp_fn(y, y_pred)
    weighted_accuracy = calculate_weighted_accuracy(tp, tn, fp, fn)
    sensitivity = calculate_sensitivity(tp, fn)
    specificity = calculate_specificity(tn, fp)
    return weighted_accuracy, sensitivity, specificity

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


def predict_bg_lines_ut_dataset(main_dir,palimpsest_name,folio_name,modality,models):
    class_name_bg = "bg_lines_ot_subtracted"
    class_name_ut = "undertext_ot_subtracted"
    palimpsest_dir = os.path.join(main_dir,palimpsest_name)
    features_bg,features_ut = read_bg_ut_features(palimpsest_dir,folio_name,class_name_ut,class_name_bg,modality)
    X = np.vstack((features_bg, features_ut))
    y = np.hstack((np.zeros(features_bg.shape[0]), np.ones(features_ut.shape[0])))
    accuracies = []
    for model in models:
        y_pred = model.predict(X)
        tp, tn, fp, fn = calculate_tp_tn_fp_fn(y, y_pred)
        accuracy = calculate_weighted_accuracy(tp, tn, fp, fn)
        accuracies.append(accuracy)
    return np.mean(accuracies)

def predict_msi_image(msi_im, models,palimpsest_dir, folio_name):
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
    ot_path = os.path.join(palimpsest_dir,folio_name,"mask",f"{folio_name}-overtext_black.png")
    ot_im = io.imread(ot_path,as_gray=True)
    if np.max(ot_im)>1:
        ot_im = ot_im/np.max(ot_im)
    ot_im[ot_im>=0.5]=1
    prediction_image[ot_im==0]=0

    return prediction_image

if __name__=="__main__":
    main_dir = r"c:\Data\PhD\palimpsest\Victor_data"
    palimpsest_name = "Paris_Coislin"
    folio_name = r"Par_coislin_393_054r"
    modality = "M"
    exp_name = "Naive_Bayesian"
    class_names_bg = ["bg_margin","bg_lines_ot_subtracted"]
    class_name_ut = "undertext_ot_subtracted"
    im_msi = read_ims_msi_object(main_dir,palimpsest_name,folio_name,modality)
    palimpsest_dir = os.path.join(main_dir,palimpsest_name)
    for class_name_bg in class_names_bg:
        box = "train"
        features_bg,features_ut = read_bg_ut_features(palimpsest_dir,folio_name,class_name_ut,class_name_bg,modality,box)
        model, accuracy_train, sensitivity_train, specificity_train = NB_train(features_bg,features_ut)
        accuracy_test, sensitivity_test, specificity_test = NB_test(main_dir,palimpsest_name,folio_name,model)
        print(
            f"Train weighted_mean_accuracy on bg {class_name_bg} train and ut {class_name_ut} is {accuracy_train} and test weighted_mean_accuracy on bg lines  and ut {class_name_ut} line is {accuracy_test}"
        )
        print(
            f"Train sensitivity on bg {class_name_bg} train and ut {class_name_ut} is {sensitivity_train} and test sensitivity on bg lines  and ut {class_name_ut} line is {sensitivity_test}"
        )
        print(
            f"Train specificity on bg {class_name_bg} train and ut {class_name_ut} is {specificity_train} and test specificity on bg lines  and ut {class_name_ut} line is {specificity_test}"
        )
        d = {f"Train weighted_mean_accuracy on bg {class_name_bg} train and ut {class_name_ut}":accuracy_train,
             f"Test Weighted_mean_accuracy on bg lines  and ut {class_name_ut} line":accuracy_test,
             f"Train sensitivity on bg {class_name_bg} train and ut {class_name_ut}":sensitivity_train,
             f"Test sensitivity on bg lines  and ut {class_name_ut} line":sensitivity_test,
             f"Train specificity on bg {class_name_bg} train and ut {class_name_ut}":specificity_train,
             f"Test specificity on bg lines  and ut {class_name_ut} line":specificity_test,}
        save_path = os.path.join(main_dir, palimpsest_name, folio_name, "miscellaneous", exp_name)
        os.makedirs(save_path, exist_ok=True)
        save_json(os.path.join(save_path, f"tain_test_{folio_name}_NB_{class_name_ut}_{class_name_bg}.json"), d)


    #pred_im = predict_msi_image(im_msi, models,palimpsest_dir, folio_name)
    #pred_im = (255 * pred_im).astype(np.uint8)
    #save_path = os.path.join(main_dir,palimpsest_name,folio_name,"miscellaneous",exp_name)
    #os.makedirs(save_path,exist_ok=True)
    #io.imsave(os.path.join(save_path,f"{folio_name}_NB_{class_name_ut}_{class_name_bg}.png"),pred_im)
    #save_json(os.path.join(save_path,f"{folio_name}_NB_{class_name_ut}_{class_name_bg}.json"),d)
