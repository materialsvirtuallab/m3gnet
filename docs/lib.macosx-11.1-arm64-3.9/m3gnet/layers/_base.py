"""
Base layer classes
"""
from typing import Callable, List, Optional

import tensorflow as tf

from m3gnet.graph import Index
from m3gnet.utils import register


@register
class GraphUpdate(tf.keras.layers.Layer):
    """
    A graph update takes a graph list representation as input and output a
    updated graph
    """

    def call(self, graph: List, **kwargs) -> List:
        """

        Args:
            graph (list): list representation of a graph
            **kwargs:
        Returns:
        """
        return graph


@register
class GraphUpdateFunc(GraphUpdate):
    """
    Update a graph with a function
    """

    def __init__(
        self,
        update_func: Callable,
        update_field: str,
        input_fields: Optional[List[str]] = None,
        **kwargs,
    ):
        """

        Args:
            update_func (Callable): A update function that is callable
            update_field (str): The graph field to update
            input_fields (list): by default only the "update_field" is the
             input to the update function. Alternatively we can specify
             multiple fields as inputs
            **kwargs:
        """
        super().__init__(**kwargs)
        self.update_func = update_func
        self.update_field: str = update_field
        if input_fields is None:
            input_fields = [update_field]  # type: ignore
        self.input_fields: List[str] = input_fields

    def call(self, graph: List, **kwargs) -> List:
        """
        Call logic of the layer
        Args:
            graph (list): list representation of the graph
            **kwargs:
        Returns: graph list
        """
        graph = graph[:]
        index = getattr(Index, self.update_field.upper())
        input_indices = [getattr(Index, i.upper()) for i in self.input_fields]
        graph[index] = self.update_func(*[graph[i] for i in input_indices])
        return graph

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns: config dict
        """
        config = super().get_config()
        config.update(
            {
                "update_func": self.update_func,
                "update_field": self.update_field,
                "input_fields": self.input_fields,
            }
        )
        return config
