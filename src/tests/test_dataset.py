import unittest

import numpy as np
from dataset import resample_nb_dataset_points


class TestDatasetMethods(unittest.TestCase):

    def test_resample_nb_dataset_points(self):
        ut_split = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([1, 1, 1])]
        nonut_split = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 1])]
        result = resample_nb_dataset_points(ut_split, nonut_split)
        self.assertEqual(result[0].shape, (6, 3))
        self.assertEqual(result[1].shape, (6,))

        ut_split = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 1])]
        nonut_split = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([1, 1, 1])]
        result = resample_nb_dataset_points(ut_split, nonut_split)
        self.assertEqual(result[0].shape, (6, 3))
        self.assertEqual(result[1].shape, (6,))

        ut_split = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([1, 1, 1])]
        nonut_split = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([1, 1, 1])]
        result = resample_nb_dataset_points(ut_split, nonut_split)
        self.assertEqual(result[0].shape, (6, 3))
        self.assertEqual(result[1].shape, (6,))


if __name__ == '__main__':
    unittest.main()
