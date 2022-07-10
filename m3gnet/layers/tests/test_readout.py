import json
import os
import unittest

import numpy as np

from m3gnet.graph import MaterialGraph
from m3gnet.layers import MultiFieldReadout, ReduceReadOut, Set2Set

CWD = os.path.abspath(os.path.dirname(__file__))


class TestReadout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(os.path.join(CWD, "data", "test_NaCl.json")) as f:
            g_list = json.load(f)
        cls.graph = MaterialGraph.from_list(g_list)
        cls.graph_w_atom_features = cls.graph.copy()
        cls.graph_w_atom_features.atoms = np.random.normal(size=(cls.graph.atoms.shape[0], 10))

    def test_set2set(self):
        set2set = Set2Set(units=3, num_steps=2, field="bonds")
        res = set2set(self.graph.as_list())
        self.assertTrue(res.shape == (1, 6))

    def test_atomreduce(self):
        ar = ReduceReadOut(method="mean", field="atoms")
        result = ar(self.graph.as_list())
        self.assertTrue(result.shape == (1, 1))

    def test_multifield(self):
        atom = ReduceReadOut(method="mean", field="atoms")
        bond = Set2Set(units=3, num_steps=2, field="bonds")
        multi = MultiFieldReadout(bond_readout=bond, atom_readout=atom)
        graph = self.graph.copy()
        graph.atoms = np.zeros_like(graph.atoms, dtype=np.float32)
        x = multi(graph.as_list())
        self.assertTrue(x.shape == (1, 7))


if __name__ == "__main__":
    unittest.main()
