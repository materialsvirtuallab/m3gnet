import unittest

import numpy as np
from pymatgen.core.structure import Lattice, Structure

from m3gnet.graph import BaseGraphConverter, MaterialGraph, RadiusCutoffGraphConverter


class TestConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    def test_base_converter(self):
        bc = BaseGraphConverter()
        g1 = bc.get_atom_features(self.s1)
        np.testing.assert_array_equal(g1, [42, 42])
        bc.set_default_states(states=[0, 0])
        state = bc.get_states(self.s1)
        self.assertListEqual(state, [0, 0])
        self.s1.states = np.array([[0, 1]])
        state = bc.get_states(self.s1)
        np.testing.assert_array_almost_equal(state.ravel(), [0, 1])

    def test_radius_cutoff(self):
        rcg = RadiusCutoffGraphConverter(cutoff=4.0, has_threebody=True, threebody_cutoff=3.0)
        g1 = rcg.convert(self.s1)
        self.assertTrue(isinstance(g1, MaterialGraph))
        self.assertTrue(g1.has_threebody)
        self.assertTrue(np.all(g1.bonds.ravel() <= 4.0))

        rcg2 = RadiusCutoffGraphConverter(cutoff=4.0, has_threebody=False, threebody_cutoff=3.0)
        g2 = rcg2.convert(self.s1)
        self.assertFalse(g2.has_threebody)


if __name__ == "__main__":
    unittest.main()
