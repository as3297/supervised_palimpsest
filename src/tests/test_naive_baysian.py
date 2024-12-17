import unittest
from miscellaneous import naive_baysian as nb


class TestNaiveBayesian(unittest.TestCase):

    def test_calculate_tp_tn_fp_fn(self):
        # Test 1: All predictions correct
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        self.assertEqual(nb.calculate_tp_tn_fp_fn(y_true, y_pred), (3, 2, 0, 0))

        # Test 2: All predictions incorrect
        y_true = [1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 0]
        self.assertEqual(nb.calculate_tp_tn_fp_fn(y_true, y_pred), (0, 0, 2, 3))

        # Test 3: Mixed correct and incorrect predictions
        y_true = [1, 0, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 1]
        self.assertEqual(nb.calculate_tp_tn_fp_fn(y_true, y_pred), (2, 1, 1, 1))

        # Test 4: No true instances, simulations for preventing division by zero
        y_true = [0, 0, 0, 0, 0]
        y_pred = [0, 0, 0, 0, 0]
        self.assertEqual(nb.calculate_tp_tn_fp_fn(y_true, y_pred), (0, 5, 0, 0))


# Run the tests
if __name__ == '__main__':
    unittest.main()
