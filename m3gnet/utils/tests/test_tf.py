import unittest

import numpy as np
import tensorflow as tf

from m3gnet.utils import (
    broadcast_states_to_atoms,
    broadcast_states_to_bonds,
    get_length,
    get_range_indices_from_n,
    get_segment_indices_from_n,
    register,
    register_plain,
    repeat_with_n,
    unsorted_segment_fraction,
    unsorted_segment_softmax,
)


class TestTF(unittest.TestCase):
    def test_calculations(self):
        np.testing.assert_array_almost_equal(get_segment_indices_from_n([2, 3]), [0, 0, 1, 1, 1])

        np.testing.assert_array_almost_equal(get_range_indices_from_n([2, 3]), [0, 1, 0, 1, 2])

        np.testing.assert_array_almost_equal(
            repeat_with_n([[0, 0], [1, 1], [2, 2]], [1, 2, 3]),
            [[0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2]],
        )

        x = np.random.normal(size=(10, 3))
        np.testing.assert_array_almost_equal(get_length(x), np.linalg.norm(x, axis=1))

    def test_segments(self):
        x = np.array([1.0, 1.0, 2.0, 3.0])
        res = unsorted_segment_fraction(x, [0, 0, 1, 1], 2)
        np.testing.assert_array_almost_equal(res, [0.5, 0.5, 0.4, 0.6])

        res = unsorted_segment_softmax(x, [0, 0, 1, 1], 2)
        np.testing.assert_array_almost_equal(res, [0.5, 0.5, 0.26894143, 0.7310586])

    def test_broadcast(self):
        from m3gnet.graph import Index

        graph = [[] for i in range(11)]
        graph[Index.STATES] = np.array([[0.0, 0.0], [1.0, 1.0]])
        graph[Index.N_BONDS] = np.array([5, 6])
        self.assertTupleEqual(tuple(broadcast_states_to_bonds(graph).shape), (11, 2))

        graph[Index.N_ATOMS] = np.array([3, 5])
        self.assertTupleEqual(tuple(broadcast_states_to_atoms(graph).shape), (8, 2))

    def test_register(self):
        class TestClass(tf.keras.layers.Layer):
            pass

        register(TestClass)
        self.assertTrue("m3gnet>TestClass" in tf.keras.utils.get_custom_objects())

        class TestClass2(tf.keras.layers.Layer):
            pass

        register_plain(TestClass2)

        self.assertTrue("TestClass2" in tf.keras.utils.get_custom_objects())


if __name__ == "__main__":
    unittest.main()
