import unittest

import numpy as np
import tensorflow as tf
from pymatgen.core.structure import Lattice, Structure

from m3gnet.graph import Index, MaterialGraph, RadiusCutoffGraphConverter


class TestConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        cls.g1 = RadiusCutoffGraphConverter(cutoff=5.0, threebody_cutoff=4.0).convert(cls.s1)

    def test_graph(self):
        self.assertTrue(isinstance(self.g1, MaterialGraph))
        glist = self.g1.as_list()
        np.testing.assert_array_almost_equal(glist[Index.ATOMS].ravel(), [42, 42])
        gstr = str(self.g1)
        self.assertTrue(gstr.startswith("<Material"))

        self.assertTrue(isinstance(self.g1.atoms, np.ndarray))
        gtf = self.g1.as_tf()
        self.assertTrue(isinstance(gtf.atoms, tf.Tensor))
        self.assertTrue(self.g1.n_atom == 2)
        self.assertTrue(self.g1.n_bond == self.g1.n_bonds[0])
        self.assertTrue(self.g1.n_struct == 1)
        self.assertTrue(self.g1.has_threebody)

        g2 = MaterialGraph.from_list(self.g1.as_list())
        self.assertTrue(self.g1 == g2)
        g3 = self.g1.copy()
        self.assertTrue(self.g1 == g3)


if __name__ == "__main__":
    unittest.main()
