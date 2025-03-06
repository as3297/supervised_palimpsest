
import matplotlib.pyplot as plt
# Set a backend explicitly (optional if not pre-configured)
matplotlib.use("TkAgg")  # Use any supported interactive backend, e.g., TkAgg or Qt5Agg
import numpy as np
from read_data import read_subset_features,read_msi_image_object
from msi_data_as_array import FragmentfromMSI_PIL,NormalizingGray

# Display spectralon target for every band
# Load non-undertext and undertext masks
main_dir = r"d:\Verona_msXL"
folio_names = ["msXL_315r_b",
"msXL_315v_b",
"msXL_318r_b",
"msXL_318v_b",
"msXL_319r_b","msXL_319v_b","msXL_322r_b","msXL_322v_b",
"msXL_323r_b","msXL_323v_b","msXL_334r_b","msXL_334v_b",
"msXL_335r_b","msXL_335v_b","msXL_344r_b","msXL_344v_b"]
modality = "M"
for folio_name in folio_names:
    print(folio_name)
    im_obj = read_msi_image_object(main_dir,folio_name,modality)
    scoords = NormalizingGray(im_obj).spectralon_coords
    scoords = np.stack(scoords)
    box = [np.min(scoords[:,0])-100,np.min(scoords[:,1])-100,np.max(scoords[:,0])+100,np.max(scoords[:,1])+100]
    fragment = FragmentfromMSI_PIL(im_obj,box).ims_img
    nb_bands = fragment.shape[-1]
    fig,ax = plt.subplots(nb_bands//2,2)
    fig.suptitle(folio_name)
    for i in range(nb_bands//2):
        ax[i,0].imshow(fragment[:,:,i],cmap="gray")
        ax[i,0].grid(False)
        ax[i, 1].imshow(fragment[:, :, i], cmap="gray")
        ax[i, 1].grid(False)
        ax[i,0].set_xticks([])  # Turn off X-axis ticks
        ax[i,0].set_yticks([])  # Turn off Y-axis ticks
        ax[i, 1].set_xticks([])  # Turn off X-axis ticks
        ax[i, 1].set_yticks([])  # Turn off Y-axis ticks

plt.show()