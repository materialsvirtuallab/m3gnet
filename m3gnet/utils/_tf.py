# -*- coding: utf-8 -*-
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

#    n = tf.shape(ns)[0]
#    max_n = tf.reduce_max(ns)
#    matrix = tf.range(n)
#    matrix = tf.tile(tf.reshape(matrix, (n, 1)), [1, max_n])
#    mask = tf.sequence_mask(ns, max_n)
#    return tf.cast(tf.boolean_mask(matrix, mask), tf.int32)


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


#    indices = get_segment_indices_from_n(n)
#    return tf.gather(tensor, indices)


def get_length(t):
    """
    equivalent to tf.linalg.norm(t, axis=1), but faster
    Args:
        t: (tf.Tensor), two dimensional tensor

    Returns:
    """
    return tf.sqrt(tf.reduce_sum(t * t, axis=1))


def _gather(data, segment_ids, counts=None, axis=0):
    return tf.gather(data, segment_ids)

#    if counts is None:
#        counts = tf.math.bincount(segment_ids)
#    return tf.repeat(data, repeats=counts, axis=axis)


def segment_softmax(data, segment_ids, weights=None, counts=None):
    """
    Segment softmax
    Args:
        data:
        segment_ids:
        weights:
        counts:

    Returns:
    """
    data = tf.cast(data, "float32")
    if weights is None:
        weights = tf.constant(1.0, dtype="float32")
    else:
        weights = tf.cast(weights, dtype="float32")
    segment_max = tf.math.segment_max(data, segment_ids)
    maxes = _gather(segment_max, segment_ids, counts)
    data -= maxes
    exp = tf.exp(data) * tf.squeeze(weights)
    softmax = tf.math.divide_no_nan(
        exp, _gather(tf.math.segment_sum(exp, segment_ids), segment_ids,
                     counts)
    )
    return tf.cast(softmax,
                   tf.keras.mixed_precision.global_policy().compute_dtype)


def unsorted_segment_softmax(data, segment_ids, num_segments, weights=None,
                             counts=None):
    """
    Segment softmax
    Args:
        data:
        segment_ids:
    Returns:
    """
    data = tf.cast(data, "float32")
    if weights is None:
        weights = tf.constant(1.0, dtype="float32")
    else:
        weights = tf.cast(weights, dtype="float32")
    segment_max = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
    maxes = _gather(segment_max, segment_ids, counts)
    data -= maxes
    exp = tf.exp(data) * tf.squeeze(weights)

    softmax = tf.math.divide_no_nan(exp, _gather(
                tf.math.unsorted_segment_sum(exp, segment_ids, num_segments),
                segment_ids,
                counts))
    return tf.cast(softmax,
                   tf.keras.mixed_precision.global_policy().compute_dtype)


def unsorted_segment_fraction(data, segment_ids, num_segments, counts=None):
    """
    Segment softmax
    Args:
        data:
        segment_ids:
    Returns:
    """
    data = tf.cast(data, "float32")
    segment_sum = tf.math.unsorted_segment_sum(data, segment_ids, num_segments)
    sums = _gather(segment_sum, segment_ids, counts)
    data = tf.math.divide_no_nan(data, sums)
    return tf.cast(data,
                   tf.keras.mixed_precision.global_policy().compute_dtype)


def append_zeros_to_match(tensor1, tensor2):
    n1 = tf.shape(tensor1)[0]
    n2 = tf.shape(tensor2)[0]
    tensor2 = tf.pad(tensor2, [[0, n1 - n2], [0, 0]])
    return tensor2


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
    #  bond_ids = get_segment_indices_from_n(graph[Index.N_BONDS])
    #  return tf.gather(graph[Index.STATES], bond_ids)


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
    # atom_ids = get_segment_indices_from_n(graph[Index.N_ATOMS])
    # return tf.gather(graph[Index.STATES], atom_ids)


def register(cls):
    return tf.keras.utils.register_keras_serializable(package="m3gnet")(cls)


def register_plain(cls):
    """register a simple class_name: instance pair to the custom object"""
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls
