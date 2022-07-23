import unittest

import numpy as np
from pymatgen.core.structure import Lattice, Molecule, Structure

from m3gnet.graph import MaterialGraph, RadiusCutoffGraphConverter
from m3gnet.layers import MLP
from m3gnet.layers._atom import GatedAtomUpdate
from m3gnet.layers._bond import ConcatAtoms
from m3gnet.layers._gn import GraphFeaturizer, GraphNetworkLayer
from m3gnet.layers._state import ConcatBondAtomState


class TestGN(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        s.states = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mol = Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])
        rc = RadiusCutoffGraphConverter(cutoff=4.0, has_threebody=True, threebody_cutoff=4.0)
        cls.struct_graph = rc.convert(s).as_list()
        cls.mol_graph = rc.convert(mol).as_list()

    def test_gn(self):
        bond_network = ConcatAtoms(neurons=[10, 9])
        atom_network = GatedAtomUpdate(neurons=[10, 5])
        state_network = ConcatBondAtomState(update_func=MLP(neurons=[5, 3]))
        gn = GraphNetworkLayer(
            bond_network=bond_network,
            atom_network=atom_network,
            state_network=state_network,
        )

        gf = GraphFeaturizer(
            n_atom_types=94,
            atom_embedding_dim=5,
            nfeat_state=5,
            rbf_type="SphericalBessel",
            max_n=3,
            max_l=3,
            cutoff=4.0,
            smooth=False,
        )

        g1 = gf(self.struct_graph)
        g2 = MaterialGraph.from_list(g1)
        self.assertTupleEqual(tuple(g2.atoms.shape), (2, 5))
        self.assertTupleEqual(tuple(g2.bonds.shape), (28, 9))
        g3 = MaterialGraph.from_list(gn(g2.as_list()))
        self.assertTupleEqual(tuple(g3.atoms.shape), (2, 5))
        self.assertTupleEqual(tuple(g3.bonds.shape), (28, 9))
        self.assertTupleEqual(tuple(g3.states.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
