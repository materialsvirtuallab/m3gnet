import unittest

import numpy as np
from pymatgen.core.structure import Lattice, Structure

from m3gnet.layers import AtomRef


class TestRef(unittest.TestCase):
    def test_ref(self):
        ar = AtomRef(property_per_element=np.random.normal(size=(94,)))
        s1 = Structure(Lattice.cubic(3), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        s2 = s1 * [1, 2, 1]
        ar.fit([s1, s2], [-10, -20])
        self.assertAlmostEqual(ar.property_per_element[42], -5.0)
        new_energies = ar.transform([s1, s2], [-10, -20])
        self.assertTrue(np.allclose(new_energies, [0, 0]))
        energies = ar.inverse_transform([s1, s2], [0, 0])
        self.assertTrue(np.allclose(energies, [-10, -20]))
        self.assertAlmostEqual(ar.predict_properties(s1 * 2), -80)


if __name__ == "__main__":
    unittest.main()
