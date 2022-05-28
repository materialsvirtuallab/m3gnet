"""
Cutoff functions
"""

import numpy as np
import tensorflow as tf


def polynomial(r: tf.Tensor, cutoff: float) -> tf.Tensor:
    """
    Polynomial cutoff function
    Args:
        r (tf.Tensor): radius distance tensor
        cutoff (float): cutoff distance

    Returns: polynomial cutoff functions

    """
    ratio = r / cutoff
    return tf.where(r <= cutoff, 1 - 6 * ratio**5 + 15 * ratio**4 - 10 * ratio**3, 0.0)


def cosine(r: tf.Tensor, cutoff: float) -> tf.Tensor:
    """
    Cosine cutoff function
    Args:
        r (tf.Tensor): radius distance tensor
        cutoff (float): cutoff distance

    Returns: cosine cutoff functions

    """
    return tf.where(r <= cutoff, tf.math.cos(np.pi * r / cutoff), 0.0)


CUTOFF_MAPPING = {"polynomial": polynomial, "cosine": cosine}
