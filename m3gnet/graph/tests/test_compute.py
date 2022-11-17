import unittest

import numpy as np
import tensorflow as tf
from pymatgen.core.structure import Lattice, Structure

from m3gnet.graph import (
    Index,
    RadiusCutoffGraphConverter,
    get_pair_vector_from_graph,
    tf_compute_distance_angle,
)


def _loop_indices(bond_atom_indices, pair_dist, cutoff=4.0):
    bin_count = np.bincount(bond_atom_indices[:, 0], minlength=bond_atom_indices[-1, 0] + 1)
    indices = []
    start = 0
    for bcont in bin_count:
        for i in range(bcont):
            for j in range(bcont):
                if start + i == start + j:
                    continue
                if pair_dist[start + i] > cutoff or pair_dist[start + j] > cutoff:
                    continue
                indices.append([start + i, start + j])
        start += bcont
    return np.array(indices)


def _calculate_cos_loop(graph, threebody_cutoff=4.0):
    """
    Calculate the cosine theta of triplets using loops
    Args:
        graph: List
    Returns: a list of cosine theta values
    """
    pair_vector = get_pair_vector_from_graph(graph)
    _, _, n_sites = tf.unique_with_counts(graph[Index.BOND_ATOM_INDICES][:, 0])
    start_index = 0
    cos = []
    for n_site in n_sites:
        for i in range(n_site):
            for j in range(n_site):
                if i == j:
                    continue
                vi = pair_vector[i + start_index].numpy()
                vj = pair_vector[j + start_index].numpy()
                di = np.linalg.norm(vi)
                dj = np.linalg.norm(vj)
                if (di <= threebody_cutoff) and (dj <= threebody_cutoff):
                    cos.append(vi.dot(vj) / np.linalg.norm(vi) / np.linalg.norm(vj))
        start_index += n_site
    return cos


class TestCompute(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        gc = RadiusCutoffGraphConverter(5)
        cls.s1 = Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        cls.s2 = Structure(Lattice.cubic(3), ["Mo", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        cls.g1 = gc.convert(cls.s1)
        cls.g2 = gc.convert(cls.s2)

    def test_compute_pair_vec(self):
        pair_vec = get_pair_vector_from_graph(self.g1.as_list())

        d = tf.linalg.norm(pair_vec, axis=1)
        _, _, _, d2 = self.s1.get_neighbor_list(r=5.0)

        np.testing.assert_array_almost_equal(np.sort(d), np.sort(d2))

    def test_compute_dist_angle(self):
        g2 = tf_compute_distance_angle(
            self.g1.as_list(),
        )
        _, _, _, d2 = self.s1.get_neighbor_list(r=5.0)

        np.testing.assert_array_almost_equal(np.sort(g2[Index.BONDS].numpy().ravel()), np.sort(d2))

        cos_loop = _calculate_cos_loop(self.g1.as_list())

        cos = g2[Index.THETA]
        np.testing.assert_array_almost_equal(cos_loop, cos)

    def test_include_threebody_indices(self):
        g3 = self.g1.as_list()[:]
        np.testing.assert_array_almost_equal(
            g3[Index.TRIPLE_BOND_INDICES],
            _loop_indices(g3[Index.BOND_ATOM_INDICES], g3[Index.BONDS].ravel()),
        )


if __name__ == "__main__":
    unittest.main()
