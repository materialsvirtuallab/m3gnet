import os
import unittest

import numpy as np
from ase import Atoms
from monty.tempfile import ScratchDir
from pymatgen.core.structure import Lattice, Molecule, Structure

from m3gnet.models import M3GNet, MolecularDynamics, Potential, Relaxer, M3GNetCalculator


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = M3GNet.load()
        cls.potential = Potential(model=cls.model)
        cls.mol = Molecule(["C", "O"], [[0, 0, 0], [1.5, 0, 0]])
        cls.structure = Structure(Lattice.cubic(3.30), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        cls.atoms = Atoms(["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]], cell=np.eye(3) * 3.30, pbc=True)
        cls.single_atoms = Structure(Lattice.cubic(6.0), ["Mo"], [[0, 0, 0]])

    def test_m3gnet(self):
        g = self.model.graph_converter(self.mol)

        val = self.model.predict_structure(self.mol).numpy().ravel()
        val_graph = self.model.predict_graph(g).numpy().ravel()

        self.assertTrue(val.size == 1)
        self.assertAlmostEqual(val, val_graph)

        vals = self.model.predict_structures([self.mol, self.mol]).numpy().ravel()
        vals_graph = self.model.predict_graphs([g, g]).numpy().ravel()

        self.assertTrue(np.allclose(vals, [val, val]))
        self.assertTrue(np.allclose(vals_graph, [val, val]))

    def test_potential(self):
        e, f, s = self.potential.get_efs(self.structure)
        self.assertAlmostEqual(e.numpy().item(), -21.3307, 3)
        self.assertTrue(np.allclose(f.numpy().ravel(), np.zeros(shape=(2, 3)).ravel(), atol=1e-3))
        self.assertTrue(
            np.allclose(
                np.diag(s[0].numpy()).ravel(),
                np.array([[28.519585, 28.519585, 28.519585]]),
                atol=1e-2,
            )
        )

    def test_single_atoms(self):
        self.potential.get_efs(self.structure)
        e, f, s = self.potential.get_efs(self.single_atoms)
        shapes = f.numpy().shape
        self.assertTupleEqual(shapes, (1, 3))

    def test_relaxer(self):
        relaxer = Relaxer()  # this loads the default model

        relax_results = relaxer.relax(self.structure)

        final_structure = relax_results["final_structure"]
        final_energy = relax_results["trajectory"].energies[-1] / 2
        self.assertAlmostEqual(final_structure.lattice.abc[0], 3.169, 2)
        self.assertAlmostEqual(final_energy.item(), -10.859, 3)

    def test_md(self):
        with ScratchDir("."):
            md = MolecularDynamics(atoms=self.structure, temperature=300, logfile="mo.log")

            md.run(10)
            self.assertTrue(os.path.isfile("mo.log"))

    def test_calculator(self):
        atoms = self.atoms.copy()
        atoms.calc = M3GNetCalculator(potential=self.potential)

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        self.assertAlmostEqual(energy, 325.37271, 3)
        self.assertTrue(
            np.allclose(
                forces[0],
                [-673.525, -673.525, -673.525],
                atol=1e-2,
            )
        )

        self.assertEqual(np.shape(energy), ())
        self.assertEqual(np.shape(forces), (2, 3))


if __name__ == "__main__":
    unittest.main()
