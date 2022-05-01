"""
Calculate distance from atom positions and indices
"""
from typing import List

import tensorflow as tf

from m3gnet.graph import get_pair_vector_from_graph
from m3gnet.utils import get_length, register


@register
class PairDistance(tf.keras.layers.Layer):
    """
    Compute pair distances from atom positions, bond indices, lattices and
    periodic offsets.
    """

    def call(self, graph: List, **kwargs):
        """
        Calculate the pair distance from a MaterialGraph.
        Args:
            graph (list): A list representation of a MaterialGraph object
            **kwargs:

        Returns: tf.Tensor distance tensor

        """
        pair_vectors = get_pair_vector_from_graph(graph)
        return get_length(pair_vectors)


@register
class PairVector(tf.keras.layers.Layer):
    """
    Compute pair atom distance vectors from graph
    """

    def call(self, graph: List, **kwargs):
        """
        Calculate the pair vector distance from a MaterialGraph.
        Args:
            graph (List): A MaterialGraph object
            **kwargs:

        Returns: tf.Tensor distance vector tensor
        """
        return get_pair_vector_from_graph(graph)
