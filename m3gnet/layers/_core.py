"""
Core layers provide basic operations, e.g., MLP
"""
from typing import Dict, List, Union

import tensorflow as tf

from m3gnet.utils import register

METHOD_MAPPING = {x: eval(f"tf.math.unsorted_segment_{x}") for x in ["sum", "prod", "max", "min", "mean"]}


@register
class Pipe(tf.keras.layers.Layer):
    """
    Simple layer for consecutive layer calls, similar to Sequential
    """

    def __init__(self, layers: List, **kwargs):
        """
        Args:
            layers（list): List of layers for consecutive calls
            **kwargs:
        """
        super().__init__(**kwargs)
        self.layers = layers

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """
        Run the inputs through the layers
        Args:
            inputs (List): a graph in list representation
            **kwargs:
        Returns: tf.Tensor
        """
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return out

    def get_config(self) -> Dict:
        """
        Get layer configuration in dictionary format
        Returns: dict

        """
        config = super().get_config()
        config["layers"] = []
        for i in self.layers:
            config["layers"].append(tf.keras.layers.serialize(i))
        return config

    @classmethod
    def from_config(cls, config: Dict) -> "Pipe":
        """
        Construct Pipe object from a config dict
        Args:
            config (dict): configuration dictionary

        Returns: Pipe object

        """
        layers_dict = config.pop("layers")
        layers = [tf.keras.layers.deserialize(i) for i in layers_dict]
        return cls(layers=layers, **config)


@register
class Embedding(tf.keras.layers.Embedding):
    """
    Thin wrapper for embedding atomic numbers into feature vectors
    """

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """
        Implementation of the layer call

        Args:
            inputs (list): list representation of graph
        Returns: tf.Tensor

        """
        return super().call(inputs[:, 0])


@register
class MLP(tf.keras.layers.Layer):
    """
    Multi-layer perceptron
    """

    def __init__(
        self,
        neurons: List[int],
        activations: Union[List, str, None] = "swish",
        kernel_regularizers: Union[List, str, None] = None,
        use_bias: bool = True,
        is_output: bool = False,
        **kwargs,
    ):
        """
        Multi-layer perceptron implementation
        Args:
            neurons (list): number of neurons in each layer
            activations (list): activation functions in each layer
            kernel_regularizers (list): regularizers, list of string or the
                ones in tf.keras.regularizers
            use_bias (bool): whether to use bias in the ANN
            is_output (bool): whether the result is the final output. If it
                is the final output, we need to make sure that it gives
                float32. This is for half-precision training
            **kwargs:
        """
        super().__init__(**kwargs)
        if isinstance(activations, str) or activations is None:
            activation_list = [activations] * len(neurons)
        else:
            activation_list = activations  # type: ignore
        if isinstance(kernel_regularizers, str) or kernel_regularizers is None:
            kernel_regularizer_list = [kernel_regularizers] * len(neurons)
        else:
            kernel_regularizer_list = kernel_regularizers  # type: ignore

        self.neurons = neurons
        self.activations: Union[List, str, None] = activations
        self.kernel_regularizers: Union[List, str, None] = kernel_regularizers

        dense_layers: List[tf.keras.layers.Layer] = [
            tf.keras.layers.Dense(i, activation=j, kernel_regularizer=reg, use_bias=use_bias)
            for i, j, reg in zip(neurons, activation_list, kernel_regularizer_list)
        ]
        if is_output:
            dense_layers.append(tf.keras.layers.Activation(None, dtype="float32", name="predictions"))

        self.pipe = Pipe(layers=dense_layers)
        self.is_output = is_output
        self.use_bias = use_bias

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """
        Implementation of the layer call

        Args:
            inputs (list): list representation of graph
        Returns: tf.Tensor

        """
        return self.pipe.call(inputs, **kwargs)

    def get_config(self) -> Dict:
        """
        Get layer configuration in dictionary format
        Returns: dict

        """
        config = super().get_config()
        config.update(
            {
                "neurons": self.neurons,
                "activations": self.activations,
                "kernel_regularizers": self.kernel_regularizers,
                "is_output": self.is_output,
                "use_bias": self.use_bias,
            }
        )
        return config


@register
class GatedMLP(tf.keras.layers.Layer):
    r"""
    Gated MLP implementation. It implements the following
        `out = MLP(x) * MLP_\\sigmoid(x)`
    where that latter changes the last layer activation function into sigmoid.
    """

    def __init__(
        self,
        neurons: List[int],
        activations: Union[List, str, None] = "swish",
        kernel_regularizers: Union[List, str, None] = None,
        use_bias: bool = True,
        **kwargs,
    ):
        """

        Args:
            neurons (list): numbrer of neurons in each layer
            activations (list): activation function in each layer
            kernel_regularizers (list): regularizer in each layer
            use_bias (bool): whether to use bias
            **kwargs:
        """
        super().__init__(**kwargs)
        if isinstance(activations, str) or activations is None:
            activation_list = [activations] * len(neurons)
        else:
            activation_list = activations[:]

        if isinstance(kernel_regularizers, str) or kernel_regularizers is None:
            kernel_regularizer_list = [kernel_regularizers] * len(neurons)
        else:
            kernel_regularizer_list = kernel_regularizers
        self.neurons = neurons
        self.activations = activations
        self.kernel_regularizers = kernel_regularizers
        dense_layers = [
            tf.keras.layers.Dense(i, activation=j, kernel_regularizer=reg, use_bias=use_bias)
            for i, j, reg in zip(neurons, activation_list, kernel_regularizer_list)
        ]
        self.pipe = Pipe(layers=dense_layers)
        activation_list[-1] = "sigmoid"
        gate_layers = [
            tf.keras.layers.Dense(i, activation=j, kernel_regularizer=reg, use_bias=use_bias)
            for i, j, reg in zip(neurons, activation_list, kernel_regularizer_list)
        ]
        self.gate = Pipe(gate_layers)
        self.use_bias = use_bias

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """
        Implementation of the layer call

        Args:
            inputs (list): list representation of graph
        Returns: tf.Tensor

        """
        return self.pipe.call(inputs, **kwargs) * self.gate.call(inputs, **kwargs)

    def get_config(self) -> Dict:
        """
        Get layer configuration in dictionary format
        Returns: dict

        """
        config = super().get_config()
        config.update(
            {
                "neurons": self.neurons,
                "activations": self.activations,
                "kernel_regularizers": self.kernel_regularizers,
                "use_bias": self.use_bias,
            }
        )
        return config
