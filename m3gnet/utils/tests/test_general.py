import unittest

import numpy as np

from m3gnet.utils import check_array_equal, check_shape_consistency, reshape_array


class TestGeneralUtils(unittest.TestCase):
    def test_check_array_equal(self):
        self.assertTrue(check_array_equal(None, None))
        self.assertFalse(check_array_equal(None, np.array([1, 2, 3])))
        self.assertFalse(check_array_equal(np.array([1, 2, 3]), None))
        self.assertTrue(check_array_equal(np.array([1, 2, 3]), np.array([1, 2, 3])))

    def test_check_shape_consistency(self):
        self.assertTrue(check_shape_consistency(np.random.normal(size=(1, 2, 3)), [1, 2, None]))
        self.assertTrue(check_shape_consistency(np.random.normal(size=(1, 2, 3)), [1, 2, 3]))
        self.assertTrue(check_shape_consistency(np.random.normal(size=(1, 2, 3)), [1, None]))

    def test_reshape(self):
        self.assertRaises(ValueError, reshape_array, np.array([1, 2, 3]), [2, None])
        self.assertEqual(reshape_array(np.array([1, 2, 3]), [3, None]).shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
