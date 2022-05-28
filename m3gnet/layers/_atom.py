"""
Edge networks.

The edge networks take a graph as inputs and calculate the updated
edge attributes
"""

from typing import List

import tensorflow as tf

from m3gnet.graph import Index
from m3gnet.utils import broadcast_states_to_bonds, register

from ._base import GraphUpdate
from ._core import GatedMLP


@register
class AtomNetwork(GraphUpdate):
    """
    Atom network that takes a graph as input and then calculate new
    atom attributes
    """

    def update_atoms(self, graph: List) -> tf.Tensor:
        """
        Take a graph input and calculate the updated atom attributes
        Args:
            graph (list): list repr of a MaterialGraph

        Returns: tf.Tensor

        """
        return graph[Index.ATOMS]

    def call(self, graph: List, **kwargs) -> List:
        """
        Args:
            graph (list): list repr of a MaterialGraph
            **kwargs:
        Returns: a new MaterialGraph in list repr

        """
        graph = graph[:]
        atoms = self.update_atoms(graph)
        graph[Index.ATOMS] = atoms
        return graph


@register
class GatedAtomUpdate(AtomNetwork):
    """
    Take the neighbor atom attributes and bond attributes, update them to the
    center atom
    """

    def __init__(self, neurons: List[int], activation: str = "swish", **kwargs):
        """
        Args:
            neurons (list): number of neurons in each layer
            activation (str): activation function
            **kwargs:
        """
        super().__init__(**kwargs)
        self.neurons = neurons
        self.bond_dense = GatedMLP(neurons=neurons, activations=[activation] * len(neurons))
        self.weight_update = tf.keras.layers.Dense(neurons[-1], activation=activation, use_bias=False)
        self.activation = activation

    def _get_reduced(self, graph: List):
        atoms_left = tf.gather(graph[Index.ATOMS], graph[Index.BOND_ATOM_INDICES][:, 0])
        atoms_right = tf.gather(graph[Index.ATOMS], graph[Index.BOND_ATOM_INDICES][:, 1])

        atoms = [atoms_left, atoms_right, graph[Index.BONDS]]

        if graph[Index.STATES] is not None:
            atoms.append(broadcast_states_to_bonds(graph))

        reduced = tf.concat(atoms, axis=-1)
        reduced = self.bond_dense(reduced) * self.weight_update(graph[Index.BOND_WEIGHTS])
        return reduced

    def update_atoms(self, graph: List) -> tf.Tensor:
        """
        Update atom attributes
        Args:
            graph (list): list repr of MaterialGraph
        Returns: tf.Tensor

        """
        reduced = self._get_reduced(graph)
        c = tf.math.unsorted_segment_sum(
            reduced,
            graph[Index.BOND_ATOM_INDICES][:, 0],
            num_segments=tf.shape(graph[Index.ATOMS])[0],
        )
        return graph[Index.ATOMS] + c

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns: dict
        """
        config = super().get_config()
        config.update(neurons=self.neurons, activation=self.activation)
        return config
