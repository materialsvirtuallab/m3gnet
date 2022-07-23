import unittest

import numpy as np
from pymatgen.core.structure import Lattice, Molecule, Structure

from m3gnet.graph import Index, RadiusCutoffGraphConverter
from m3gnet.layers import PairRadialBasisExpansion


class TestAgg(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        mol = Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])
        rc = RadiusCutoffGraphConverter(cutoff=4.0, has_threebody=True, threebody_cutoff=4.0)
        cls.struct_graph = rc.convert(s).as_list()
        cls.mol_graph = rc.convert(mol).as_list()

    def test_bond_network(self):
        prb = PairRadialBasisExpansion(rbf_type="Gaussian", centers=np.linspace(0, 4, 10), width=0.5)
        graph = prb(self.struct_graph)
        self.assertTupleEqual(tuple(graph[Index.BONDS].shape), (28, 10))
        prb = PairRadialBasisExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        graph = prb(self.struct_graph)
        self.assertTupleEqual(tuple(graph[Index.BONDS].shape), (28, 9))


if __name__ == "__main__":
    unittest.main()
