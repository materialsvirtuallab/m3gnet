"""Aggregate classes. Aggregating happens when bond attributes """
from typing import Callable, List, Union

import tensorflow as tf

from m3gnet.graph import Index
from m3gnet.utils import get_segment_indices_from_n, register

from ._core import METHOD_MAPPING


@register
class AtomReduceState(tf.keras.layers.Layer):
    """
    Reduce atom attributes to states via sum or mean
    """

    def __init__(self, method: Union[Callable, str] = "mean", **kwargs):
        """

        Args:
            method (str or Callable): the method for the aggregation,
                choose from ["sum", "prod", "max", "min", "mean"]
                default to "mean"
            **kwargs:
        """
        super().__init__(**kwargs)
        self.method = method
        if isinstance(method, str):
            if method not in METHOD_MAPPING:
                raise ValueError("Unrecognized method name")
            method_func = METHOD_MAPPING[method]
        else:
            method_func = method
        self.method_func: Callable = method_func

    def call(self, graph: List, **kwargs) -> tf.Tensor:
        """
        Main layer logic
        Args:
            graph (list): input graph list representation
            **kwargs:
        Returns:
        """
        atom_segment = get_segment_indices_from_n(graph[Index.N_ATOMS])
        return self.method_func(
            graph[Index.ATOMS],
            atom_segment,
            num_segments=tf.shape(graph[Index.STATES])[0],
        )

    def get_config(self) -> dict:
        """
        Get the configuration dict
        Returns:

        """
        config = super().get_config()
        config.update({"method": self.method})
        return config
