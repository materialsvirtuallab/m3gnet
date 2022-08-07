"""
Edge networks.

The edge networks take a graph as inputs and calculate the updated
edge attributes
"""

from typing import List

import tensorflow as tf

from m3gnet.graph import Index
from m3gnet.utils import broadcast_states_to_bonds, get_segment_indices_from_n, register

from ._base import GraphUpdate
from ._core import GatedMLP
from ._basis import RBF_ALLOWED, RadialBasisFunctions


@register
class BondNetwork(GraphUpdate):
    """
    Edge network that takes a graph as input and then calculate new
    bond attributes
    """

    def update_bonds(self, graph: List) -> tf.Tensor:
        """
        Update bond info in the graph
        Args:
            graph (list): list representation of a graph
        Returns: tf.Tensor
        """
        return graph[Index.BONDS]

    def call(self, graph: List, **kwargs) -> List:
        """
        Update the bond and return a copy of the graph
        Args:
            graph (list): graph list representation
            **kwargs:
        Returns: graph list representation
        """
        graph = graph[:]
        bonds = self.update_bonds(graph)
        graph[Index.BONDS] = bonds
        return graph


def _unity_weights(graph):
    return tf.ones_like(graph[Index.BONDS][:, :1])


def _bonds_to_weights(graph):
    return graph[Index.BONDS]


@register
class PairRadialBasisExpansion(BondNetwork):
    """
    Expand the radial distance onto a basis
    """

    def __init__(self, rbf_type: str = "Gaussian", **kwargs):
        """

        Args:
            rbf_type (str): RBF type, choose from "Gaussian" and
            "SphericalBessel"
            **kwargs (dict): the necessary parameters for initialize the RBF
        """
        self.rbf_type = rbf_type
        rbf_kwds = {}
        keys = []
        for k in kwargs.keys():
            if k in RBF_ALLOWED[rbf_type]["params"]:  # type: ignore
                keys.append(k)
        for k in keys:
            rbf_kwds.update({k: kwargs.pop(k)})
        self.rbf_kwds = rbf_kwds
        self.rbf = RadialBasisFunctions(rbf_type=rbf_type, **rbf_kwds)
        if rbf_type == "Gaussian":
            self.weight_func = _unity_weights
        else:
            self.weight_func = _bonds_to_weights
        super().__init__(**kwargs)

    def update_bonds(self, graph: List) -> tf.Tensor:
        """
        Update the bond info using RBF
        Args:
            graph (list): list representation of a graph

        Returns: updated bond info
        """
        return self.rbf(graph[Index.BONDS][:, 0])  # noqa

    def call(self, graph: List, **kwargs):
        """
        On top of the changing bond, also add bond weight for later use
        Args:
            graph (list): list representation of a graph
            **kwargs:

        Returns:
        """
        graph = graph[:]
        bonds = self.update_bonds(graph)
        graph[Index.BONDS] = bonds
        graph[Index.BOND_WEIGHTS] = self.weight_func(graph)
        return graph

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns: dict
        """
        config = super().get_config()
        config.update({"rbf_type": self.rbf_type})
        config.update(**self.rbf_kwds)
        return config


@register
class ConcatAtoms(BondNetwork):
    r"""
    .. math::
        eij^\prime = Update(vi⊕vj⊕eij⊕u)
    """

    def __init__(self, neurons: List[int], activation: str = "swish", **kwargs):
        """
        Concatenate the atom attributes and bond attributes (and optionally
        the state attributes, and then pass to an update function, e.g., an MLP

        Args:
            neurons (list): number of neurons in each layer
            activation (str): activation function
        """
        self.neurons = neurons
        self.activation = activation
        self.update_func = GatedMLP(neurons=neurons, activations=[activation] * len(neurons))
        self.weight_func = tf.keras.layers.Dense(neurons[-1], use_bias=False)
        super().__init__(**kwargs)

    def update_bonds(self, graph: List, **kwargs) -> tf.Tensor:
        """
        Update bond information
        Args:
            graph (list): list representation of the graph
            **kwargs:
        Returns:
        """
        n_bonds = tf.shape(graph[Index.BOND_ATOM_INDICES])[0]
        atoms = tf.reshape(tf.gather(graph[Index.ATOMS], graph[Index.BOND_ATOM_INDICES]), (n_bonds, -1))
        if graph[Index.STATES] is None:
            concat = tf.concat([atoms, graph[Index.BONDS]], axis=-1)
        else:
            states = broadcast_states_to_bonds(graph)
            concat = tf.concat([atoms, graph[Index.BONDS], states], axis=-1)
        return self.update_func(concat) * self.weight_func(graph[Index.BOND_WEIGHTS]) + graph[Index.BONDS]

    def get_config(self) -> dict:
        """
        get config dict for serialization
        Returns: dict
        """
        config = super().get_config()
        config.update({"neurons": self.neurons, "activation": self.activation})
        return config


@register
class ThreeDInteraction(tf.keras.layers.Layer):
    """
    Include 3D interactions to the bond update
    """

    def __init__(self, update_network: tf.keras.layers.Layer, update_network2: tf.keras.layers.Layer, **kwargs):
        """
        Args:
            update_network (tf.keras.layers.Layer): keras layer for update
                the atom attributes before merging with 3d interactions
            update_network2 (tf.keras.layers.Layer): keras layer for update
                the bond information after merging with 3d interactions
            **kwargs:
        """
        super().__init__(**kwargs)
        self.update_network = update_network
        self.update_network2 = update_network2

    def call(self, graph: List, three_basis: tf.Tensor, three_cutoff: float, **kwargs) -> List:
        """

        Args:
            graph (list): graph list representation
            three_basis (tf.Tensor): three body basis expansion
            three_cutoff (float): cutoff radius
            **kwargs:
        Returns:
        """
        graph = graph[:]
        end_atom_index = tf.gather(graph[Index.BOND_ATOM_INDICES][:, 1], graph[Index.TRIPLE_BOND_INDICES][:, 1])
        atoms = self.update_network(graph[Index.ATOMS])
        atoms = tf.gather(atoms, end_atom_index)
        basis = three_basis * atoms
        n_bonds = tf.reduce_sum(graph[Index.N_BONDS])
        weights = tf.reshape(tf.gather(three_cutoff, graph[Index.TRIPLE_BOND_INDICES]), (-1, 2))
        weights = tf.math.reduce_prod(weights, axis=-1)
        basis = basis * weights[:, None]
        new_bonds = tf.math.unsorted_segment_sum(
            basis,
            get_segment_indices_from_n(graph[Index.N_TRIPLE_IJ]),
            num_segments=n_bonds,
        )
        graph[Index.BONDS] = graph[Index.BONDS] + self.update_network2(new_bonds)
        return graph

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns: dict
        """
        config = super().get_config()
        config.update(
            {
                "update_network": self.update_network,
                "update_network2": self.update_network2,
                "out_unit": self.out_unit,
            }
        )
        return config
