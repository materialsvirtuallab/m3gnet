import json
import os
import unittest

import numpy as np
import tensorflow as tf
from m3gnet.models import M3GNet, Potential
from m3gnet.trainers import PotentialTrainer, Trainer
from monty.tempfile import ScratchDir
from pymatgen.core import Structure

DIR = os.path.dirname(os.path.abspath(__file__))


class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(os.path.join(DIR, "Li-O.json")) as f:
            data = json.load(f)

        cls.structures = []
        cls.energies = []
        cls.bandgaps = []
        for j in data.values():
            cls.structures.append(Structure.from_dict(j["structure"]))
            cls.energies.append(j["energy"])
            cls.bandgaps.append(j["band_gap"])

    def test_train_with_state_and_batches(self):
        """
        This error only occurs when include_states=True and the batch size
        is less than the number of data points you are training on (which
        is typically the case)
        """
        m3gnet = M3GNet(n_blocks=1, units=5, is_intensive=True, include_states=True)
        trainer = Trainer(model=m3gnet, optimizer=tf.keras.optimizers.Adam(1e-2))

        # 50 > 32
        trainer.train(self.structures[:50], self.bandgaps[:50], batch_size=32, epochs=2, train_metrics=["mae"])
        self.assertTrue(m3gnet.predict_structures(self.structures[:2]).numpy().shape == (2, 1))

    def test_train_bandgap(self):
        m3gnet = M3GNet(n_blocks=1, units=5, is_intensive=True)
        trainer = Trainer(model=m3gnet, optimizer=tf.keras.optimizers.Adam(1e-2))

        trainer.train(self.structures[:30], self.bandgaps[:30], epochs=2, train_metrics=["mae"])
        self.assertTrue(m3gnet.predict_structures(self.structures[:2]).numpy().shape == (2, 1))

    def test_train_energy(self):
        m3gnet = M3GNet(n_blocks=1, units=5, is_intensive=False)

        trainer = Trainer(model=m3gnet, optimizer=tf.keras.optimizers.Adam(1e-3))

        with ScratchDir("."):
            trainer.train(
                self.structures[:30],
                self.energies[:30],
                epochs=2,
                train_metrics=["mae"],
            )
        self.assertTrue(m3gnet.predict_structures(self.structures[:2]).numpy().shape == (2, 1))

    def test_train_potential(self):
        m3gnet = M3GNet(n_blocks=1, units=5, is_intensive=False)
        potential = Potential(model=m3gnet)

        trainer = PotentialTrainer(potential=potential, optimizer=tf.keras.optimizers.Adam(1e-3))

        n_atoms = [len(i) for i in self.structures]
        fake_forces = [np.random.normal(size=(i, 3)) for i in n_atoms]
        fake_stress = [np.random.normal(size=(3, 3)) for i in n_atoms]
        with ScratchDir("."):
            trainer.train(
                self.structures[:30],
                self.energies[:30],
                fake_forces[:30],
                fake_stress[:30],
                validation_graphs_or_structures=self.structures[30:40],
                val_energies=self.energies[30:40],
                val_forces=fake_forces[30:40],
                val_stresses=fake_stress[30:40],
                epochs=2,
                fit_per_element_offset=True,
                save_checkpoint=False,
            )
        self.assertTrue(m3gnet.predict_structures(self.structures[:2]).numpy().shape == (2, 1))

    def test_train_energy_offset(self):
        m3gnet = M3GNet(n_blocks=1, units=5, is_intensive=False)

        trainer = Trainer(model=m3gnet, optimizer=tf.keras.optimizers.Adam(1e-3))

        trainer.train(
            self.structures[:30],
            self.energies[:30],
            epochs=2,
            train_metrics=["mae"],
            fit_per_element_offset=True,
        )
        self.assertTrue(m3gnet.predict_structures(self.structures[:2]).numpy().shape == (2, 1))


if __name__ == "__main__":
    unittest.main()
