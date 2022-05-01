"""
Common metrics used in M3GNet
"""
from typing import Callable

import tensorflow as tf

METRICS = {"rmse": tf.keras.metrics.RootMeanSquaredError()}

REVERSE_NAME_MAPPING = {"mean_absolute_error": "mae"}

MONITOR_MAPPING = {"val_AUC": "max", "val_mae": "min", "val_loss": "min"}


def _get_metric(metric):
    try:
        return tf.keras.metrics.get(metric)
    except ValueError:
        if isinstance(metric, Callable):
            return metric

        if metric.lower() not in METRICS:
            raise ValueError("metric not found")
        return METRICS[metric.lower()]


def _get_metric_string(metric):
    if isinstance(metric, tf.keras.metrics.Metric):
        return metric.__class__.__name__
    name = metric.__name__
    if name in REVERSE_NAME_MAPPING:
        return REVERSE_NAME_MAPPING[name]
    return name
