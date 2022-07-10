"""
Tensorflow related utility
"""
from typing import List

import tensorflow as tf


def get_segment_indices_from_n(ns: tf.Tensor) -> tf.Tensor:
    """
    Get segment indices from number array. For example if
    ns = [2, 3], then the function will return [0, 0, 1, 1, 1]

    Args:
        ns: tf.Tensor, the number of atoms/bonds array

    Returns:
        object:

    Returns: segment indices tensor
    """
    return tf.repeat(tf.range(tf.shape(ns)[0]), repeats=ns)


def get_range_indices_from_n(ns: tf.Tensor) -> tf.Tensor:
    """
    Give ns = [2, 3], return [0, 1, 0, 1, 2]
    Args:
        ns: tf.Tensor, the number of atoms/bonds array

    Returns: range indices
    """
    max_n = tf.reduce_max(ns)
    n = tf.shape(ns)[0]
    n_range = tf.range(max_n)
    matrix = tf.tile(n_range[None, ...], [n, 1])
    mask = tf.sequence_mask(ns, max_n)
    return tf.cast(tf.boolean_mask(matrix, mask), tf.int32)


def repeat_with_n(tensor: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
    """
    Repeat the first dimension according to n array.
    The
    Args:
        tensor: Tensor, tensor to augment
        n: Tensor, array to specify the repeat times for
            each item in the first dimension of tensor

    Returns: repeated tensor

    """
    return tf.repeat(tensor, repeats=n, axis=0)


def get_length(t):
    """
    equivalent to tf.linalg.norm(t, axis=1), but faster
    Args:
        t: (tf.Tensor), two dimensional tensor

    Returns: tf.Tensor
    """
    return tf.sqrt(tf.reduce_sum(t * t, axis=1))


def unsorted_segment_softmax(data, segment_ids, num_segments, weights=None):
    """
    Unsorted segment softmax with optional weights
    Args:
        data (tf.Tensor): original data
        segment_ids (tf.Tensor): tensor segment ids
        num_segments (int): number of segments
    Returns: tf.Tensor
    """
    data = tf.cast(data, "float32")
    if weights is None:
        weights = tf.constant(1.0, dtype="float32")
    else:
        weights = tf.cast(weights, dtype="float32")

    segment_max = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
    maxes = tf.gather(segment_max, segment_ids)
    data -= maxes
    exp = tf.exp(data) * tf.squeeze(weights)

    softmax = tf.math.divide_no_nan(
        exp,
        tf.gather(tf.math.unsorted_segment_sum(exp, segment_ids, num_segments), segment_ids),
    )
    return tf.cast(softmax, tf.keras.mixed_precision.global_policy().compute_dtype)


def unsorted_segment_fraction(data, segment_ids, num_segments):
    """
    Segment softmax
    Args:
        data (tf.Tensor): original data
        segment_ids (tf.Tensor): segment ids
        num_segments (tf.Tensor): number of segments
    Returns:
    """
    data = tf.cast(data, "float32")
    segment_sum = tf.math.unsorted_segment_sum(data, segment_ids, num_segments)
    sums = tf.gather(segment_sum, segment_ids)
    data = tf.math.divide_no_nan(data, sums)
    return tf.cast(data, tf.keras.mixed_precision.global_policy().compute_dtype)


def broadcast_states_to_bonds(graph: List):
    """
    Broadcast state attributes of shape [Ns, Nstate] to
    bond attributes shape [Nb, Nstate]
    Args:
        graph: List

    Returns: broadcasted state attributes

    """
    from m3gnet.graph import Index

    return tf.repeat(graph[Index.STATES], graph[Index.N_BONDS], axis=0)


def broadcast_states_to_atoms(graph: List):
    """
    Broadcast state attributes of shape [Ns, Nstate] to
    bond attributes shape [Na, Nstate]
    Args:
        graph: List

    Returns: broadcasted state attributes

    """
    from m3gnet.graph import Index

    return tf.repeat(graph[Index.STATES], graph[Index.N_ATOMS], axis=0)


def register(cls):
    """
    Register m3gnet classes
    Args:
        cls: class name
    Returns:
    """
    return tf.keras.utils.register_keras_serializable(package="m3gnet")(cls)


def register_plain(cls):
    """register a simple class_name: instance pair to the custom object

    Args:
        cls: class name
    """
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls
