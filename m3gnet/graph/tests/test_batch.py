import unittest

import numpy as np
from pymatgen.core.structure import Lattice, Structure

from m3gnet.graph import (
    MaterialGraph,
    MaterialGraphBatch,
    MaterialGraphBatchEnergyForceStress,
    RadiusCutoffGraphConverter,
    assemble_material_graph,
)


class TestBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        gc = RadiusCutoffGraphConverter(5)
        s1 = Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        s2 = Structure(Lattice.cubic(3), ["Mo", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        cls.g1 = gc.convert(s1)
        cls.g2 = gc.convert(s2)

    def test_assemble_material_graph(self):
        g_assembled = assemble_material_graph([self.g1, self.g2])
        n_bond1 = self.g1.bonds.shape[0]
        n_bond2 = self.g2.bonds.shape[0]
        n_atom1 = self.g1.atoms.shape[0]

        self.assertTrue(g_assembled.atoms.shape == (4, 1))
        self.assertTrue(g_assembled.bonds.shape[0] == n_bond1 + n_bond2)

        self.assertTrue(np.allclose(g_assembled.bond_atom_indices[:n_bond1], self.g1.bond_atom_indices))
        self.assertTrue(
            np.allclose(
                g_assembled.bond_atom_indices[n_bond1:],
                self.g2.bond_atom_indices + n_atom1,
            )
        )

        g_assembled2 = assemble_material_graph([self.g1.as_list(), self.g2.as_list()])

        g_assembled_mg = MaterialGraph.from_list(g_assembled2)
        self.assertTrue(g_assembled_mg.atoms.shape == (4, 1))
        self.assertTrue(g_assembled_mg.bonds.shape[0] == n_bond1 + n_bond2)
        self.assertTrue(np.allclose(g_assembled_mg.bond_atom_indices[:n_bond1], self.g1.bond_atom_indices))
        self.assertTrue(
            np.allclose(
                g_assembled_mg.bond_atom_indices[n_bond1:],
                self.g2.bond_atom_indices + n_atom1,
            )
        )

    def test_sequential_assemble(self):
        # test stepwise combine
        g3 = assemble_material_graph([self.g1, self.g2])
        g3 = assemble_material_graph([self.g1, g3])

        g4 = assemble_material_graph([self.g1, self.g1, self.g2])
        self.assertTrue(g3 == g4)

    def test_sequential_assemble_list(self):
        g1 = self.g1.as_list()
        g2 = self.g2.as_list()

        g3 = assemble_material_graph([g1, g2])
        g3 = assemble_material_graph([g1, g3])

        g4 = assemble_material_graph([g1, g1, g2])
        self.assertTrue(MaterialGraph.from_list(g3) == MaterialGraph.from_list(g4))

    def test_materialgraph_batch(self):
        graphs = [self.g1, self.g2]
        targets = [0.1, 0.2]
        gb = MaterialGraphBatch(graphs, targets, batch_size=2)
        first_batch = gb[0]
        self.assertTrue(isinstance(first_batch[0], MaterialGraph))
        self.assertTrue(first_batch[1].shape == (2,))
        index = gb.graph_index[:]
        gb.on_epoch_end()
        index2 = gb.graph_index[:]
        self.assertTrue(set(index) == set(index2))
        self.assertTrue(len(gb) == 1)

    def test_materialgraph_efs(self):
        graphs = [self.g1, self.g2]
        energies = [0.1, 0.2]
        forces = [np.random.normal(size=(2, 3)), np.random.normal(size=(2, 3))]
        stresses = [np.random.normal(size=(3, 3)), np.random.normal(size=(3, 3))]
        mgb = MaterialGraphBatchEnergyForceStress(
            graphs=graphs,
            energies=energies,
            forces=forces,
            stresses=stresses,
            batch_size=2,
        )
        first_batch = mgb[0]
        outshapes = [i.shape for i in first_batch[1]]
        self.assertListEqual(
            list(outshapes[0]),
            [
                2,
            ],
        )
        self.assertListEqual(list(outshapes[1]), [4, 3])
        self.assertListEqual(list(outshapes[2]), [2, 3, 3])


if __name__ == "__main__":
    unittest.main()
