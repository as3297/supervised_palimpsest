import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from joblib import Parallel
from src.miscellaneous.calc_nn_ut import find_distance_btw_feat


class TestCalcNnUt(unittest.TestCase):
    def setUp(self):
        self.features_ut_same = np.array([[1, 2], [3, 4]])
        self.features_page_same = np.array([[1, 2], [3, 4]])
        self.xs_page = np.array([9, 10])
        self.ys_page = np.array([11, 12])
        self.xs_ut = np.array([1, 1])
        self.ys_ut = np.array([1, 2])
        self.n = 2
        self.same_page = False
        self.nb_processes = 1
        self.chunk_size = 1

    def test_find_distance_btw_feat(self):
        same_page = False
        dist_chunk, xs_chunk, ys_chunk = find_distance_btw_feat(self.features_ut_same, self.features_page_same, self.xs_page,
                                                                self.ys_page, self.n, same_page, nb_processes=1, chunk_size=2)
        print(dist_chunk)
        print(xs_chunk)
        print(ys_chunk)

    def test_find_distance_btw_feat_normal_case(self):
        with patch("joblib.Parallel", return_value=MagicMock()) as mock_parallel:
            mock_parallel.return_value.__enter__.return_value = Parallel(self.nb_processes)
            distance, xs, ys = find_distance_btw_feat(self.features_ut_same, self.features_page_same, self.xs_page, self.ys_page,
                                                      self.n, self.same_page, self.nb_processes, self.chunk_size)
            self.assertEqual(distance.shape[1], self.n)
            self.assertEqual(xs.shape[1], self.n)
            self.assertEqual(ys.shape[1], self.n)




if __name__ == "__main__":
    #unittest.main()
    test = TestCalcNnUt()
    test.test_find_distance_btw_feat()