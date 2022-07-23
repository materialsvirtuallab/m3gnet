"""
Training graph network property models
"""
import logging
import os
import platform
from glob import glob
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from ase import Atoms
from pymatgen.core.structure import Molecule, Structure

from m3gnet.callbacks import ManualStop
from m3gnet.graph import MaterialGraph, MaterialGraphBatch
from m3gnet.layers import AtomRef
from m3gnet.models import M3GNet

from ._metrics import MONITOR_MAPPING, _get_metric, _get_metric_string

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
PLATFORM = platform.platform()


class Trainer:
    """
    Trainer for material properties
    """

    def __init__(self, model: M3GNet, optimizer: tf.keras.optimizers.Optimizer):
        """
        Args:
            model (M3GNet): a M3GNet model
            optimizer (tf.keras.optimizers.Optimizer): a keras optimizer
        """
        self.model = model
        self.optimizer = optimizer
        self.initial_epoch = 0

    def restart_from_directory(self, dirname: str):
        """
        Continue previous model training from a directory
        Args:
            dirname (str): directory name
        Returns:
        """
        filenames = glob(os.path.join(dirname, "*.index"))
        latest_model = sorted(filenames, key=os.path.getctime)[-1]
        latest_model = latest_model.rsplit(".", 1)[0]
        self.initial_epoch = int(latest_model.split("/", 1)[1].split("-", 1)[0]) + 1
        self.model.load_weights(latest_model)

    def train(
        self,
        graphs_or_structures: List,
        targets: List,
        validation_graphs_or_structures: List = None,
        validation_targets: List = None,
        loss: tf.keras.losses.Loss = tf.keras.losses.MSE,
        train_metrics: Optional[List] = None,
        val_metrics: Optional[List] = None,
        val_monitor: str = "val_mae",
        batch_size: int = 128,
        epochs: int = 1000,
        callbacks: List = None,
        save_checkpoint: bool = True,
        verbose: int = 1,
        clip_norm: Optional[float] = 10.0,
        fit_per_element_offset: bool = False,
    ):
        """
        Args:
            graphs_or_structures (list): a list of MaterialGraph or structures
            targets (list): list of properties in float
            validation_graphs_or_structures (list): optional list of validation
                graphs or structures
            validation_targets (list): optional list of properties
            loss (tf.keras.losses.Loss): loss object
            train_metrics (list): list of train metrics
            val_metrics (list): list of validation metrics
            val_monitor (str): field to monitor during validation, e.g.,
                "val_mae", "val_acc" or "val_auc"
            batch_size (int): batch size of combining graphs
            epochs (int): epochs for training the data
            callbacks (list): list of callback functions
            save_checkpoint (bool): whether to save model check point
            verbose (bool): whether to show model training progress
            clip_norm (float): gradient norm clip
            fit_per_element_offset (bool): whether to train an element-wise
                offset, e.g., elemental energies etc. If trained, such energy
                will be summed to the neural network predictions.

        Returns: None
        """

        if isinstance(graphs_or_structures[0], MaterialGraph):
            graphs = graphs_or_structures
        elif isinstance(graphs_or_structures[0], (Structure, Molecule, Atoms)):
            graphs = [self.model.graph_converter(i) for i in graphs_or_structures]
        else:
            raise ValueError("Graph types not recognized")

        if fit_per_element_offset:
            ar = AtomRef(max_z=self.model.n_atom_types + 1)
            ar.fit(graphs, targets)
            self.model.set_element_refs(ar.property_per_element)

        val_metrics = val_metrics or ["mae"]
        mgb = MaterialGraphBatch(graphs, targets, batch_size=batch_size)

        if train_metrics is not None:
            train_metrics = [_get_metric(metric) for metric in train_metrics]
            train_metric_names = [f"{_get_metric_string(i) }" for i in train_metrics]
            has_train_metrics = True
        else:
            train_metric_names = []
            has_train_metrics = False

        if val_metrics is not None:
            val_metrics = [_get_metric(metric) for metric in val_metrics]
            val_metric_names = [f"val_{_get_metric_string(i)}" for i in val_metrics]
            has_val_metrics = True
        else:
            val_metric_names = []
            has_val_metrics = False

        if validation_graphs_or_structures is not None and validation_targets is not None:
            has_validation = True
            if isinstance(validation_graphs_or_structures[0], MaterialGraph):
                validation_graphs = validation_graphs_or_structures
            elif isinstance(validation_graphs_or_structures[0], (Structure, Molecule, Atoms)):
                validation_graphs = [self.model.graph_converter(i) for i in validation_graphs_or_structures]
            else:
                raise ValueError("Graph types not recognized")

            mgb_val = MaterialGraphBatch(
                validation_graphs,
                validation_targets,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            has_validation = False

        if callbacks is None:
            callbacks = [tf.keras.callbacks.History()]

        callbacks.append(ManualStop())

        if has_validation and save_checkpoint:
            if val_monitor not in val_metric_names:
                raise ValueError(f"val_monitor {val_monitor} not in the " f"val_metric_names {val_metric_names}")

            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"callbacks/{{epoch:05d}}-{{{val_monitor}:.6f}}",
                    monitor=val_monitor,
                    save_weights_only=True,
                    save_best_only=True,
                    mode=MONITOR_MAPPING[val_monitor],
                )
            )

        if verbose:
            pbar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            pbar.set_params({"verbose": verbose, "epochs": epochs})
            callbacks.append(pbar)

        if len(callbacks) > len({i.__class__ for i in callbacks}):
            logger.warning("Duplicated callbacks found")

        callback_list = tf.keras.callbacks.CallbackList(callbacks)
        callback_list.on_train_begin()
        callback_list.set_model(self.model)
        stop_training = False

        @tf.function(experimental_relax_shapes=True)
        def predict(model, graph_batch):
            return model(graph_batch)

        @tf.function(experimental_relax_shapes=True)
        def train_one_step(model, graph_list, target_list):
            with tf.GradientTape() as tape:
                pred_list: tf.Tensor = model(graph_list)
                loss_val = loss(target_list, pred_list)
            if "macOS" in PLATFORM and "arm64" in PLATFORM and tf.config.list_physical_devices("GPU"):
                # This is a workaround for a bug in tensorflow-metal that fails when tape.gradient is called.
                with tf.device("/cpu:0"):
                    grads = tape.gradient(loss_val, model.trainable_variables)
            else:
                grads = tape.gradient(loss_val, model.trainable_variables)
            return loss_val, grads, pred_list

        for epoch in range(self.initial_epoch, epochs):
            callback_list.on_epoch_begin(epoch=epoch, logs={"epoch": epoch})
            epoch_loss_avg = tf.keras.metrics.Mean()

            if has_val_metrics:
                val_metric_values = {}
            if has_train_metrics:
                train_metric_values = {}

            train_predictions: Union[list, tf.Tensor] = []
            train_targets: Union[list, tf.Tensor] = []

            for batch_index, batch in enumerate(mgb):
                callback_list.on_batch_begin(batch=batch_index)
                graph_batch, target_batch = batch
                if isinstance(graph_batch, MaterialGraph):
                    graph_batch = graph_batch.as_list()
                if isinstance(target_batch, np.ndarray) and target_batch.ndim == 1:
                    target_batch = target_batch.reshape((-1, 1))

                loss_val, grads, predictions = train_one_step(self.model, graph_batch, target_batch)
                global_norm = tf.linalg.global_norm(grads)

                if clip_norm is not None:
                    grads, _ = tf.clip_by_global_norm(grads, clip_norm, use_norm=global_norm)

                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                epoch_loss_avg.update_state(loss_val)
                logs = {"loss": epoch_loss_avg.result().numpy()}

                callback_list.on_batch_end(batch=batch_index, logs=logs)
                train_predictions.append(predictions)
                train_targets.append(target_batch)

                if getattr(self.model, "stop_training", False):
                    stop_training = True
                    break

            if stop_training:
                break

            epoch_log = {"loss": epoch_loss_avg.result().numpy()}
            train_predictions = tf.concat(train_predictions, axis=0)
            train_targets = tf.concat(train_targets, axis=0)
            if has_train_metrics:
                for index, metric_name in enumerate(train_metric_names):
                    train_metric_values[metric_name] = train_metrics[index](  # type: ignore
                        train_targets.numpy().ravel(), train_predictions.numpy().ravel()
                    )
                epoch_log.update(
                    **{metric_name: train_metric_values[metric_name].numpy() for metric_name in train_metric_names}
                )

            if has_validation:
                val_predictions = []
                val_targets = []
                for val_index, batch in enumerate(mgb_val):
                    graph_batch, target_batch = batch
                    if isinstance(graph_batch, MaterialGraph):
                        graph_batch = graph_batch.as_list()
                    if target_batch.ndim == 1:
                        target_batch = target_batch.reshape((-1, 1))
                    predictions = predict(self.model, graph_batch)
                    val_predictions.append(predictions)
                    val_targets.append(target_batch)
                val_predictions = tf.concat(val_predictions, axis=0)
                val_targets = tf.concat(val_targets, axis=0)
                if has_val_metrics:
                    for index, metric_name in enumerate(val_metric_names):
                        val_metric_values[metric_name] = val_metrics[index](
                            val_targets.numpy().ravel(),  # type: ignore
                            val_predictions.numpy().ravel(),  # type: ignore
                        )
                    val_logs = {metric_name: val_metric_values[metric_name].numpy() for metric_name in val_metric_names}
                    epoch_log.update(**val_logs)
            callback_list.on_epoch_end(epoch=epoch, logs=epoch_log)
            mgb.on_epoch_end()
        callback_list.on_train_end()

        if save_checkpoint and has_validation:
            best_model_weights = sorted(glob("callbacks/*.index"), key=os.path.getctime)[-1]
            best_model_weights = best_model_weights.rsplit(".", 1)[0]
            self.model.load_weights(best_model_weights)
            self.model.save("best_model")
