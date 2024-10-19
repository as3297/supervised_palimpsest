import os
import unittest
from unittest import mock

import numpy as np
from miscellaneous.rdx_alg import apply_RDX


class TestRDXAlg(unittest.TestCase):
    def setUp(self):
        self.main_dir = os.path.dirname(__file__)
        self.palimpsest_name = 'test_palimpsest'
        self.folio_name = 'test_folio'
        self.modality = 'test_modality'

    @mock.patch('miscellaneous.rdx_alg.read_msi_image_object')
    @mock.patch('miscellaneous.rdx_alg.FullImageFromPILImageCube')
    @mock.patch('miscellaneous.rdx_alg.read_page_coord')
    @mock.patch('miscellaneous.rdx_alg.read_subset_features')
    @mock.patch('miscellaneous.rdx_alg.calc_Mahalanobis_dist')
    def test_apply_RDX(self, mock_dist, mock_features, mock_coord, mock_image_obj, mock_pil):
        mock_pil.return_value = np.zeros((10, 10, 3))
        mock_image_obj.return_value = mock.Mock(ims_img=np.zeros((10, 10, 3)))
        mock_coord.return_value = (1, 9, 1, 9)
        mock_features.return_value = (np.zeros((10, 3)), None, None)
        mock_dist.return_value = np.ones((8, 8))

        result = apply_RDX(self.main_dir, self.palimpsest_name, self.folio_name, self.modality)
        np.testing.assert_array_equal(result, np.ones((8, 8)))

    @mock.patch('miscellaneous.rdx_alg.read_msi_image_object')
    @mock.patch('miscellaneous.rdx_alg.FullImageFromPILImageCube')
    @mock.patch('miscellaneous.rdx_alg.read_page_coord')
    @mock.patch('miscellaneous.rdx_alg.read_subset_features')
    @mock.patch('miscellaneous.rdx_alg.calc_Mahalanobis_dist')
    def test_apply_RDX_different_shapes(self, mock_dist, mock_features, mock_coord, mock_image_obj, mock_pil):
        mock_pil.return_value = np.zeros((20, 20, 3))
        mock_image_obj.return_value = mock.Mock(ims_img=np.zeros((20, 20, 3)))
        mock_coord.return_value = (1, 19, 1, 19)
        mock_features.return_value = (np.zeros((20, 3)), None, None)
        mock_dist.return_value = np.ones((18, 18))

        result = apply_RDX(self.main_dir, self.palimpsest_name, self.folio_name, self.modality)
        np.testing.assert_array_equal(result, np.ones((18, 18)))


if __name__ == "__main__":
    unittest.main()
