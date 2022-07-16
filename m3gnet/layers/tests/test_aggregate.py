import unittest

import numpy as np

from m3gnet.graph import Index
from m3gnet.layers import AtomReduceState


class TestAgg(unittest.TestCase):
    def test_agg(self):
        graph = [[] for _ in range(13)]
        graph[Index.ATOMS] = np.random.normal(size=(10, 3))
        graph[Index.N_ATOMS] = np.array([5, 5])
        graph[Index.STATES] = np.random.normal(size=(2, 5))
        ars = AtomReduceState(method="mean")
        res = ars(graph)
        self.assertTrue(np.allclose(res[0], np.mean(graph[Index.ATOMS][:5], axis=0), atol=1e-5))
        self.assertTrue(np.allclose(res[1], np.mean(graph[Index.ATOMS][5:], axis=0), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
