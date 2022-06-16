"""
state networks

The global state networks take a graph as inputs and calculate the updated
global attributes
"""

from typing import Callable, List

import tensorflow as tf

from m3gnet.graph import Index
from m3gnet.utils import register

from ._base import GraphUpdate


@register
class StateNetwork(GraphUpdate):
    """
    Edge network that takes a graph as input and then calculate new
    bond attributes
    """

    def update_states(self, graph: List) -> tf.Tensor:
        """
        Calculate the new state attributes
        Args:
            graph (list): list repr of a MaterialGraph

        Returns: tf.Tensor
        """
        return graph[Index.STATES]

    def call(self, graph: List, **kwargs) -> List:
        """
        Args:
            graph (list): list repr of a MaterialGraph
            **kwargs:
        Returns: new graph in list repr

        """
        graph = graph[:]
        states = self.update_states(graph)
        graph[Index.STATES] = states
        return graph


@register
class ConcatBondAtomState(StateNetwork):
    r"""
    u^\prime = Update(\bar e^\prime⊕\bar v^\prime⊕u)
    """

    def __init__(self, update_func: Callable, bond_agg_func: Callable = None, atom_agg_func: Callable = None, **kwargs):
        """
        Args:
            update_func (callable): the core update function
            bond_agg_func (callable): function to aggregate bond to state
            atom_agg_func (callable): function to aggregate atom to state
            **kwargs:
        """
        self.update_func = update_func
        self.bond_agg_func = bond_agg_func
        self.atom_agg_func = atom_agg_func
        super().__init__(**kwargs)

    def update_states(self, graph: List) -> tf.Tensor:
        """
        Args:
            graph (list): list repr of a MaterialGraph
        Returns:
        """
        info_list = [graph[Index.STATES]]
        if self.bond_agg_func is not None:
            bond_agg = self.bond_agg_func(graph)
            info_list.append(bond_agg)
        if self.atom_agg_func is not None:
            atom_agg = self.atom_agg_func(graph)
            info_list.append(atom_agg)

        if len(info_list) > 1:
            concat = tf.concat(info_list, axis=-1)
        else:
            concat = info_list[0]
        return self.update_func(concat)

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns: dict
        """
        config = super().get_config()
        config.update(
            {
                "update_func": self.update_func,
                "bond_agg_func": self.bond_agg_func,
                "atom_agg_func": self.atom_agg_func,
            }
        )
        return config
