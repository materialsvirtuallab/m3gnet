"""
General utility
"""
from typing import Optional, Sequence

import numpy as np


def check_array_equal(array1: Optional[np.ndarray], array2: Optional[np.ndarray], rtol: float = 1e-3) -> bool:
    """
    Check the equality of two arrays
    Args:
        array1 (np.ndarray): first array
        array2 (np.narray): second array
        rtol (float): relative tolerance
    Returns: Bool
    """
    if array1 is None and array2 is None:
        return True
    if (array1 is None) ^ (array2 is None):
        return False
    return np.allclose(array1, array2, rtol=rtol)


def check_shape_consistency(array: Optional[np.ndarray], shape: Sequence) -> bool:
    """
    Check if array complies with shape. Shape is a sequence of
    integer that may end with None. If None is at the end of shape,
    then any shapes in array after that dimension will match with shape.

    Example: array with shape [10, 20, 30, 40] matches with [10, 20, None], but
        does not match with shape [10, 20, 30, 20]

    Args:
        array (np.ndarray or None): array to be checked
        shape (Sequence): integer array shape, it may ends with None
    Returns: bool
    """
    if array is None:
        return True
    if all(i is None for i in shape):
        return True

    array_shape = array.shape
    valid_dims = [i for i in shape if i is not None]
    n_for_check = len(valid_dims)
    return all(i == j for i, j in zip(array_shape[:n_for_check], valid_dims))


def reshape_array(array: np.ndarray, shape: Sequence) -> np.ndarray:
    """
    Take an array and reshape it according to shape. Here shape may contain
    None field at the end.

    if array shape is [3, 4] and shape is [3, 4, None], then array is shaped
    to [3, 4, 1]. If the two shapes do not match then report an error

    Args:
        array (np.ndarray): array to be reshaped
        shape (Sequence): shape dimensions

    Returns: np.ndarray, reshaped array
    """
    if not check_shape_consistency(array, shape):
        raise ValueError("array cannot be reshaped due to mismatch")

    if array.ndim >= len(shape):
        return array
    shape_r = [i if i is not None else 1 for i in shape]
    missing_dim = range(len(array.shape), len(shape_r))
    array_r = np.expand_dims(array, axis=list(missing_dim))
    tiles = [i // j for i, j in zip(shape_r, array_r.shape)]
    return np.tile(array_r, tiles)
