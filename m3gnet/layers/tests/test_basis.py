import unittest

import numpy as np

from m3gnet.layers import RadialBasisFunctions


class TestRBF(unittest.TestCase):
    def test_rbf(self):
        r = np.linspace(1, 5, 11)
        rbf_gaussian = RadialBasisFunctions(rbf_type="Gaussian", centers=np.linspace(0, 5, 10), width=0.5)
        rbf = rbf_gaussian(r)
        self.assertTupleEqual(tuple(rbf.shape), (11, 10))
        rbf_sb = RadialBasisFunctions(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=5.0, smooth=False)
        rbf = rbf_sb(r)
        self.assertTupleEqual(tuple(rbf.shape), (11, 9))

        rbf_sb = RadialBasisFunctions(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=5.0, smooth=True)
        rbf = rbf_sb(r)
        self.assertTupleEqual(tuple(rbf.shape), (11, 3))


if __name__ == "__main__":
    unittest.main()
