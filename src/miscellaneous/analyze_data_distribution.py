from dataset import read_features_labels

def test_data_load():
  EPOCHS = 35
  osp = os.path.join
  main_dir = r"C:\Data\PhD\palimpsest\Victor_data"
  band_list_path = osp(main_dir, "band_list.txt")
  bands = read_band_list(band_list_path)
  bands = sublist_of_bands(bands, "M")
  max_val_path = osp(main_dir, "bands_max_val.json")
  max_vals = read_max_vals(max_val_path, bands)
  image_dir_path = osp(main_dir, r"msXL_315r_rotated")
  bbox_fpath = osp(image_dir_path, "dataset_split.json")
  folioname = r"msXL_315r_b"
  ut_mask_path = osp(image_dir_path, "mask", "msXL_315r_b-undertext_black.png")
  nonut_mask_path = osp(image_dir_path, "mask", r"msXL_315r_b-not_undertext.png")
  im_msi_pil_ob = ImageCubePILobject(image_dir_path, folioname, bands, 0)
  bbox_dict = read_json(bbox_fpath)
  trainset_ut, trainset_nonut = read_features_labels(bbox_dict, im_msi_pil_ob,
                                                     ut_mask_path, nonut_mask_path, max_vals, "train")
  from matplotlib import pyplot as plt
  plt.figure("Undertext")
  for el in range(0, len(trainset_ut[1]), 1000):
    plt.plot(trainset_ut[0][el])
  plt.figure("Not undertext")
  for el in range(0, len(trainset_nonut[1]), 1000):
    plt.plot(trainset_nonut[0][el])
  plt.show()
  for epoch in range(EPOCHS):
    print("Epoch",epoch)
    nb_nonut_train_samples = len(trainset_nonut[1])
    nb_ut_train_samples = len(trainset_ut[1])
    print("Nb of train nonut samples",nb_nonut_train_samples)
    print("Nb of train ut samples", nb_ut_train_samples)
    trainset_nonut = shuffle_dataset_split(trainset_nonut)
    trainset = equalize_nb_dataset_points(trainset_ut,trainset_nonut)
    trainset = shuffle_dataset_split(trainset)
    nb_train_samples = len(trainset[1])
    print("Nb of train samples", nb_train_samples)