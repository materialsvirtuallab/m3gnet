import unittest

import numpy as np

from m3gnet.graph import Index
from m3gnet.layers import GraphUpdateFunc


class TestAgg(unittest.TestCase):
    def test_agg(self):
        graph = [[] for _ in range(13)]
        graph[Index.ATOMS] = np.random.normal(size=(10, 3))
        graph[Index.N_ATOMS] = np.array([5, 5])
        graph[Index.STATES] = np.random.normal(size=(2, 5))
        guf = GraphUpdateFunc(update_func=lambda x: x * 100, update_field="atoms")
        graph2 = guf(graph)
        self.assertTrue(np.allclose(graph2[Index.ATOMS], graph[Index.ATOMS] * 100))


if __name__ == "__main__":
    unittest.main()
