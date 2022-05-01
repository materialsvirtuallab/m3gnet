"""
Callback functions
"""
import os
from typing import Dict

import tensorflow as tf


class ManualStop(tf.keras.callbacks.Callback):
    """
    Stop the training manually by putting a "STOP" file in the directory
    """

    def on_batch_end(self, epoch: int, logs: Dict = None) -> None:
        """
        Codes called at the end of a batch
        Args:
            epoch (int): epoch id
            logs (Dict): log dict

        Returns: None

        """
        if os.path.isfile("STOP"):
            self.model.stop_training = True
