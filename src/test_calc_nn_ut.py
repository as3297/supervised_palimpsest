import unittest

import numpy as np
from src.miscellaneous.calc_nn_ut import find_distance_btw_feat


class TestFindDistanceBtwFeat(unittest.TestCase):

    def setUp(self):
        self.features_ut = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]], dtype=np.float32)
        self.xs_ut = np.array([1, 2], dtype=np.float32)
        self.ys_ut = np.array([1, 2], dtype=np.float32)
        self.features_page = np.array([[2, 3, 4], [5, 6, 7],[7,8,9]], dtype=np.float32)
        self.xs_page = np.array([2, 3], dtype=np.float32)
        self.ys_page = np.array([2, 3], dtype=np.float32)
        self.n = 2
        self.same_page = False

    def test_find_distance_btw_feat(self):
        result = find_distance_btw_feat(
            self.features_ut, self.xs_ut, self.ys_ut,
            self.features_page, self.xs_page, self.ys_page,
            self.n, self.same_page
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(set(result.keys()), {"dist", "xs_ut", "ys_ut", "xs", "ys"})

        for value in result.values():
            self.assertIsInstance(value, np.ndarray)

    def test_same_page(self):
        dist_dict = find_distance_btw_feat(
            self.features_ut, self.xs_ut, self.ys_ut,
            self.features_page, self.xs_page, self.ys_page,
            self.n, True
        )
        self.assertIsInstance(dist_dict, dict)
        self.assertGreater(len(dist_dict["dist"]), len(self.features_ut))


unittest.main()
