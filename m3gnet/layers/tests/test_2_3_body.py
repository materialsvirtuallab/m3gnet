import unittest

import numpy as np
from pymatgen.core.structure import Lattice, Molecule, Structure

from m3gnet.graph import RadiusCutoffGraphConverter, tf_compute_distance_angle
from m3gnet.layers import PairDistance, PairVector, SphericalBesselWithHarmonics


class TestTwoBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        mol = Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])
        rc = RadiusCutoffGraphConverter(cutoff=4.0, has_threebody=True, threebody_cutoff=4.0)
        cls.struct_graph = rc.convert(s).as_list()
        cls.mol_graph = rc.convert(mol).as_list()

    def test_twobody(self):
        pv = PairVector()
        self.assertTrue(pv(self.struct_graph).shape == (28, 3))
        self.assertTrue(pv(self.mol_graph).shape == (2, 3))

        pd = PairDistance()
        self.assertTrue(pd(self.struct_graph).shape == (28,))
        self.assertTrue(np.allclose(pd(self.mol_graph), np.array([1.1, 1.1])))

    def test_threebody(self):
        sph = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=4.0)
        sarray = sph(tf_compute_distance_angle(self.struct_graph))
        self.assertTrue(sarray.shape[1] == 9)
        marray = sph(tf_compute_distance_angle(self.mol_graph))
        self.assertTrue(marray.shape[1] == 9)


if __name__ == "__main__":
    unittest.main()
