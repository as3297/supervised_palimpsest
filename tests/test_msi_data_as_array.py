import numpy as np
import os, sys
# Add the root directory (one level up) to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
# Assuming the required classes and methods are found in the specified path
from src.msi_data_as_array import PatchesfromMSI_PIL, FragmentfromMSI_PIL
from src.pil_image_cube import ImageCubePILobject
from matplotlib import pyplot as plt
from src.util import read_band_list


class TestPatchesfromMSI_PIL():
    def __init__(self,palimpsest_name,folio_name,modalities,nb_points):

        self.win = 100
        band_list = read_band_list(os.path.join(palimpsest_name,folio_name,"band_list.txt"),modalities)
        self.pil_msi_obj = ImageCubePILobject(palimpsest_name, folio_name, band_list, 0)
        self.nb_bands = self.pil_msi_obj.nb_bands
        self.does_it_align(nb_points)



    def does_it_align(self,nb_points):
        init_point = (1200, 2200)
        hw = self.win // 2
        points_coord = [(init_point[0] + i*(hw*2+1), init_point[1]) for i in range(nb_points)]
        frag_coord = [init_point[0]-hw,init_point[1]-hw,init_point[0]+nb_points*(2*hw+1)-hw,init_point[1]+(2*hw+1)-hw]
        patches_msi = PatchesfromMSI_PIL(self.pil_msi_obj, points_coord, self.win).ims_imgs
        max_val = np.max(patches_msi)

        # Concatenating patches into one big image
        patches_msi_concat = np.zeros(((hw*2+1),(hw*2+1)*nb_points,self.nb_bands))

        for (x, y), patch in zip(points_coord, patches_msi):
            x = x - init_point[0]
            y = y - init_point[1]
            patches_msi_concat[y:y+2*hw+1, x:x+2*hw+1,:] = patch
        frag_msi = FragmentfromMSI_PIL(self.pil_msi_obj , frag_coord).ims_img

        np.testing.assert_array_equal(
            patches_msi_concat,
            frag_msi,
            err_msg="Patches and fragments do not align"
        )
        print("Test concluded")
        patches_msi_concat = patches_msi_concat/max_val
        plt.figure()
        plt.imshow(patches_msi_concat[:,:,-4:-1])
        plt.show()





if __name__ == '__main__':
    root_dir = r"D:"
    palimpsest_name = "Verona_msXL"
    main_data_dir = os.path.join(root_dir, palimpsest_name)
    folio_name = "msXL_335v_b"
    modality = ["M"]
    nb_points = 10
    TestPatchesfromMSI_PIL(main_data_dir,folio_name,modality,nb_points)

