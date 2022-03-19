# -*- coding: utf-8 -*-
import unittest
import numpy as np

from m3gnet.layers import RadialBasisFunctions


class TestRBF(unittest.TestCase):
    def test_rbf(self):
        r = np.linspace(1, 5, 11)
        rbf_gaussian = RadialBasisFunctions(rbf_type="Gaussian",
                                            centers=np.linspace(0, 5, 10),
                                            width=0.5)
        rbf = rbf_gaussian(r)
        self.assertTupleEqual(rbf.shape, (11, 10))

        rbf_sb = r


if __name__ == "__main__":
    unittest.main()

