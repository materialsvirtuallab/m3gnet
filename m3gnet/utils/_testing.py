# -*- coding: utf-8 -*-
"""
Testing suit
"""
import tensorflow as tf


def assert_nested_equal(item1, item2) -> bool:
    """
    Check nested array/list to be equal
    Args:
        item1: array-like
        item2: array-like

    Returns: true or false

    """

    if item1.__class__ != item2.__class__:
        return False

    if isinstance(item1, dict):
        if len(item1.keys()) != len(item2.keys()):
            return False
        if any(
                [not assert_nested_equal(item1[key], item2[key]) for key in
                 item1.keys()]
        ):
            return False
    elif isinstance(item1, (tuple, list, range)):
        if len(item1) != len(item1):
            return False
        if any([not assert_nested_equal(i, j) for i, j in zip(item1, item2)]):
            return False
    elif isinstance(item1, float):
        return abs(item1 - item2) < 1e-8 * item1
    # captures set, bool, str, int

    return item1 == item2


def layer_equal(layer1, layer2):
    config1 = layer1.get_config()
    config2 = layer2.get_config()
    if set(config1.keys()) != set(config2.keys()):
        return False
    equal = True
    for k in config1.keys():
        item1 = config1[k]
        item2 = config2[k]
        if isinstance(item1, tf.keras.layers.Layer):
            if not layer_equal(item1, item2):
                return False
        if not assert_nested_equal(item1, item2):
            return False
    return equal

