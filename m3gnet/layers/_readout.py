"""
Readout compress a graph into a vector
"""
from typing import List, Optional

import tensorflow as tf

from m3gnet.config import DataType
from m3gnet.graph import Index
from m3gnet.utils import (
    get_segment_indices_from_n,
    register,
    unsorted_segment_fraction,
    unsorted_segment_softmax,
)

from ._core import METHOD_MAPPING, MLP


@register
class ReadOut(tf.keras.layers.Layer):
    """
    Readout reduces a graph into a tensor
    """

    def call(self, graph: List, **kwargs) -> tf.Tensor:
        """
        Args:
            graph (list): list repr of a MaterialGraph
            **kwargs:
        Returns: tf.Tensor, tensor readout

        """
        raise NotImplementedError


@register
class MultiFieldReadout(ReadOut):
    """
    Read both bond and atom
    """

    def __init__(
        self,
        bond_readout: Optional[ReadOut] = None,
        atom_readout: Optional[ReadOut] = None,
        include_states: bool = True,
        merge: Optional[tf.keras.layers.Layer] = None,
        **kwargs,
    ):
        """

        Args:
            bond_readout (ReadOut): the bond readout instance
            atom_readout (ReadOut: the atom readout instance
            include_states (bool): whether to include states
            merge (tf.keras.layers.Layer): method to merge different readout
            **kwargs:
        """
        self.bond_readout = bond_readout
        self.atom_readout = atom_readout
        self.include_states = include_states
        self.merge = merge or tf.keras.layers.Concatenate(axis=-1)
        super().__init__(**kwargs)

    def call(self, graph: List, **kwargs) -> tf.Tensor:
        """
        Args:
            graph (list): list repr of a MaterialGraph
            **kwargs:
        Returns:

        """
        features = []
        if self.bond_readout is not None:
            features.append(self.bond_readout(graph))

        if self.atom_readout is not None:
            features.append(self.atom_readout(graph))

        if self.include_states and graph[Index.STATES] is not None:
            features.append(graph[Index.STATES])

        if len(features) > 1:
            return self.merge(features)
        return features[0]

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns: dict
        """
        config = super().get_config()
        config.update(
            {
                "bond_readout": self.bond_readout,
                "atom_readout": self.atom_readout,
                "include_states": self.include_states,
                "merge": self.merge,
            }
        )
        return config


@register
class ReduceReadOut(ReadOut):
    """
    Reduce atom or bond attributes into lower dimensional tensors as readout.
    This could be summing up the atoms or bonds, or taking the mean, etc.
    """

    def __init__(self, method: str = "mean", field="atoms", **kwargs):
        """
        Args:
            method (str): method for the reduction
            field (str): the field of MaterialGraph to perform the reduction
            **kwargs:
        """
        self.method = method
        self.field = field
        super().__init__(**kwargs)
        self.method_func = METHOD_MAPPING.get(method)

    def call(self, graph: List, **kwargs) -> tf.Tensor:
        """
        Args:
            graph (list): list repr of a MaterialGraph
            **kwargs:
        Returns:

        """
        field = graph[getattr(Index, self.field.upper())]
        n_field = graph[getattr(Index, f"n_{self.field}".upper())]
        return self.method_func(  # type: ignore
            field,
            get_segment_indices_from_n(n_field),
            num_segments=tf.shape(n_field)[0],
        )

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns: dict
        """
        config = super().get_config()
        config.update({"method": self.method, "field": self.field})
        return config


@register
class Set2Set(ReadOut):
    """
    The Set2Set readout function
    """

    def __init__(self, units: int, num_steps: int, field: str, **kwargs):
        """
        Args:
            units (int): number of neurons in the set2set layer
            num_steps (int): number of LSTM steps
            field (str): the field of MaterialGraph to perform the readout
            **kwargs:
        """

        super().__init__(**kwargs)
        self.units = units
        self.num_steps = num_steps
        self.lstm = tf.keras.layers.LSTM(units=units, stateful=False, return_state=True)
        self.dense = tf.keras.layers.Dense(units=units)
        self.field = field

    def call(self, graph: List, **kwargs) -> tf.Tensor:
        """
        Args:
            graph (list): list repr of a MaterialGraph
            **kwargs:
        Returns:

        """
        field = graph[getattr(Index, self.field.upper())]
        field = self.dense(field)
        n_struct = tf.shape(graph[Index.N_ATOMS])[0]
        counts = graph[getattr(Index, f"n_{self.field}".upper())]
        indices = get_segment_indices_from_n(counts)
        q_star = tf.zeros([n_struct, 2 * self.units], dtype=DataType.tf_float)
        h = [
            tf.zeros([n_struct, self.units], dtype=DataType.tf_float),
            tf.zeros([n_struct, self.units], dtype=DataType.tf_float),
        ]
        for _ in range(self.num_steps):
            qt, state_h, state_c = self.lstm(tf.expand_dims(q_star, axis=1), initial_state=h)
            h = [state_h, state_c]
            eit = field * tf.repeat(qt, repeats=counts, axis=0)
            ait = unsorted_segment_softmax(eit, indices, num_segments=n_struct)
            rt = tf.math.unsorted_segment_sum(ait * field, indices, num_segments=n_struct)
            q_star = tf.concat([qt, rt], axis=-1)
        return q_star

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns: dict
        """
        d = super().get_config()
        d.update({"units": self.units, "num_steps": self.num_steps, "field": self.field})
        return d


@register
class WeightedReadout(ReadOut):
    """
    Perform a weighted average of the readout field. Weights are learnable
    from this layer
    """

    def __init__(self, neurons: List[int], activation="swish", field: str = "atoms", **kwargs):
        """

        Args:
            neurons (list): list of number of neurons in each layer
            activation (str): activation function
            field (str): the field to perform the readout, choose from
                "atoms" or "bonds"
            **kwargs:
        """
        super().__init__(**kwargs)
        n_layer = len(neurons)
        self.mlp = MLP(neurons=neurons, activations=[activation] * n_layer)
        neurons = neurons[:-1] + [1]  # type: ignore
        acts = [activation] * (n_layer - 1) + ["sigmoid"]
        self.weight = MLP(neurons=neurons, activations=acts)
        self.neurons = neurons
        self.activation = activation
        self.field = field

    def call(self, graph: List, **kwargs) -> tf.Tensor:
        """
        Args:
            graph (list): list repr of a MaterialGraph
            **kwargs:
        Returns:
        """
        field = graph[getattr(Index, self.field.upper())]
        n_field = graph[getattr(Index, f"n_{self.field}".upper())]

        updated_field = self.mlp(field)
        weights = self.weight(field)
        indices = get_segment_indices_from_n(n_field)
        n_structs = tf.shape(n_field)[0]
        factor = unsorted_segment_fraction(weights[:, 0], indices, num_segments=n_structs)
        return tf.math.segment_sum(factor[:, None] * updated_field, indices)

    def get_config(self):
        """
        Get config dict for serialization
        Returns: dict
        """
        config = super().get_config()
        config.update(
            {
                "neurons": self.neurons,
                "activation": self.activation,
                "field": self.field,
            }
        )
        return config
