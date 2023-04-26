"""
M3GNet potential trainer
"""
from typing import List, Optional
import numpy as np
import json
import tensorflow as tf
from ase import Atoms
from pymatgen.core import Structure, Molecule
import datetime
import sys
import os
from m3gnet.callbacks import ManualStop
from m3gnet.graph import Index
from m3gnet.graph import MaterialGraph
from m3gnet.graph import MaterialGraphBatchEnergyForceStress
from m3gnet.layers import AtomRef
from m3gnet.models import Potential


class PotentialTrainer:
    """
    Trainer for M3GNet potential
    """
    def __init__(self, potential: Potential, optimizer: tf.keras.optimizers.Optimizer):
        """
        Args:
            potential (M3GNet): a M3GNet model
            optimizer (tf.keras.optimizers.Optimizer): a keras optimizer
        """
        self.potential = potential
        self.optimizer = optimizer

    def train(
        self,
        graphs_or_structures: List,
        energies: List,
        forces: List,
        stresses: Optional[List] = None,
        validation_graphs_or_structures: List = None,
        val_energies: List = None,
        val_forces: List = None,
        val_stresses: List = None,
        loss: tf.keras.losses.Loss = tf.keras.losses.MSE,
        force_loss_ratio: float = 1,
        stress_loss_ratio: float = 0.1,
        batch_size: int = 32,
        epochs: int = 1000,
        callbacks: List = None,
        save_checkpoint: bool = True,
        early_stop_patience: int = 200,
        verbose: int = 1,
        fit_per_element_offset: bool = False,
        data_dir = '',
    ):
        """
        Args:
            graphs_or_structures (list): a list of MaterialGraph or structures
            energies (list): list of train energies
            forces (list): list of train forces
            stresses (list): list of train stresses
            validation_graphs_or_structures (list): optional list of validation
                graphs or structures
            val_energies (list): list of val energies
            val_forces (list): list of val forces
            val_stresses (list): list of val stresses
            loss (tf.keras.losses.Loss): loss object
            force_loss_ratio (float): the ratio of forces in loss
            stress_loss_ratio (float): the ratio of stresses in loss
            train_metrics (list): list of train metrics
            val_metrics (list): list of validation metrics
            val_monitor (str): field to monitor during validation, e.g.,
                "val_mae", "val_acc" or "val_auc"
            batch_size (int): batch size of combining graphs
            epochs (int): epochs for training the data
            callbacks (list): list of callback functions
            save_checkpoint (bool): whether to save model check point
            early_stop_patience (int): patience for early stop
            verbose (bool): whether to show model training progress
            fit_per_element_offset (bool): whether to train an element-wise
                offset, e.g., elemental energies etc. If trained, such energy
                will be summed to the neural network predictions.

        Returns: None
        """

        if isinstance(graphs_or_structures[0], MaterialGraph):
            graphs = graphs_or_structures
        elif isinstance(graphs_or_structures[0], (Structure, Molecule, Atoms)):
            graphs = [
                self.potential.model.graph_converter(i) for i in graphs_or_structures
            ]
        else:
            raise ValueError("Graph types not recognized")

        if fit_per_element_offset:
            ar = AtomRef(max_z=self.potential.model.n_atom_types + 1)
            ar.fit(graphs, energies)
            self.potential.model.set_element_refs(ar.property_per_element)

        mgb = MaterialGraphBatchEnergyForceStress(
            graphs,
            energies=energies,
            forces=forces,
            stresses=stresses,
            batch_size=batch_size,
        )

        if validation_graphs_or_structures is not None and val_energies is not None:
            has_validation = True
            if isinstance(validation_graphs_or_structures[0], MaterialGraph):
                validation_graphs = validation_graphs_or_structures
            elif isinstance(
                validation_graphs_or_structures[0], (Structure, Molecule, Atoms)
            ):
                validation_graphs = [
                    self.potential.model.graph_converter(i)
                    for i in validation_graphs_or_structures
                ]
            else:
                raise ValueError("Graph types not recognized")

            mgb_val = MaterialGraphBatchEnergyForceStress(
                validation_graphs,
                energies=val_energies,
                forces=val_forces,
                stresses=val_stresses,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            has_validation = False

        has_stress = stresses is not None

        def _flat_loss(x, y):
            return loss(tf.reshape(x, (-1,)), tf.reshape(y, (-1,)))
        
        def _flat_loss_stress(x, y):
            
            mask2d = tf.constant(np.array([[1,1,0],[1,1,0],[0,0,0]]))
            mask2d = tf.cast(mask2d, tf.float32)
            print("mask2d: {}".format(mask2d))
            print("X: {}".format(x))
            print("Y: {}".format(y))
            
            mulx = x*mask2d
            muly = y*mask2d

            print("Mul x: {}".format(mulx))
            print("Mul y: {}".format(muly))

            print("Type y: {}".format(type(mulx)))
            print("Type x: {}".format(type(muly)))
          
            return loss(mulx, muly) 
           # return loss(tf.math.mul(x, mask2d), tf.math.mul(y, mask2d)) 
        
        def _mae(x, y):
            x = tf.reshape(x, tf.shape(y))
            return tf.reduce_mean(tf.math.abs(x - y))

        def _loss(target_batch, graph_pred_batch, n_atoms):
            n_atoms_temp = tf.cast(n_atoms, dtype=target_batch[0].dtype)
            e_target = target_batch[0] / n_atoms_temp
            e_target = e_target[:, None]
            e_pred = graph_pred_batch[0] / n_atoms_temp[:, None]
            e_loss = _flat_loss(e_target, e_pred)
            f_loss = _flat_loss(target_batch[1], graph_pred_batch[1])
            e_metric = _mae(e_target, e_pred)
            f_metric = _mae(target_batch[1], graph_pred_batch[1])

            s_loss = 0
            s_metric = 0
            if has_stress:
                # Changed _flat_loss to _flat_loss_stress to accomodate the 2d mask
                s_loss = _flat_loss_stress(target_batch[2], graph_pred_batch[2])
                #s_loss = _flat_loss(target_batch[2], graph_pred_batch[2])
                
                mask2d = tf.constant([[1,1,0], [1,1,0], [0,0,0]])
                mask2d = tf.cast(mask2d, tf.float32)
                #print(type(target_batch[2]))
                s_metric = _mae(target_batch[2]*mask2d, graph_pred_batch[2]*mask2d)
                #s_metric = _mae(tf.math.mul(target_batch[2], mask_2d), tf.math.mul(graph_pred_batch[2], mask_2d))
                #s_metric = _mae(target_batch[2], graph_pred_batch[2])
            return (
                e_loss + force_loss_ratio * f_loss + stress_loss_ratio * s_loss,
                e_metric,
                f_metric,
                s_metric,
            )

        if callbacks is None:
            callbacks = [tf.keras.callbacks.History()]

        if verbose:
            pbar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            pbar.set_params({"verbose": verbose, "epochs": epochs})
            callbacks.append(pbar)

        callbacks.append(ManualStop())
        N_GPU = os.getenv('CUDA_VISIBLE_DEVICES')
        dir_name = "checkpoints/"+str(N_GPU)+"_f_t-{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.datetime.now())
        print(dir_name)
        os.makedirs(dir_name)
        #print(os.getcwd())

        with open(dir_name+'/train_conf.json', 'w') as log:
            log.write(json.dumps({'batch_size':batch_size, 'force_loss_ratio':force_loss_ratio, 'stress_loss_ratio':stress_loss_ratio, 'early_stop_patience':early_stop_patience, 'fit_per_element_offset':fit_per_element_offset, 'data_dir':data_dir}))
          
        # json.dump({'batch_size':batch_size, 'force_loss_ratio':force_loss_ratio, 'stress_loss_ratio':stress_loss_ratio, 'early_stop_patience':early_stop_patience, 'fit_per_element_offset':fit_per_element_offset, 'data_dir':data_dir},  open(dir_name+'/train_conf.json'))
        if has_validation and save_checkpoint:
            name_temp = (dir_name + "/{epoch:05d}-{val_MAE:.6f}-"
                "{val_MAE(E):.6f}-{val_MAE(F):.6f}"
            )
            if has_stress:
                name_temp += "-{val_MAE(S):.6f}"
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=name_temp,
                    monitor="val_MAE",
                    save_weights_only=False,
#                    save_best_only=True,
                    mode="min",
                )
            )

        if early_stop_patience:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor="val_MAE", patience=200)
            )

        callback_list = tf.keras.callbacks.CallbackList(callbacks)
        callback_list.on_train_begin()
        callback_list.set_model(self.potential.model)

        @tf.function(experimental_relax_shapes=True)
        def train_one_step(potential, graph_list, target_list):
            with tf.GradientTape() as tape:
                if has_stress:
                    pred_list = potential.get_efs_tensor(
                        graph_list, include_stresses=True
                    )
                else:
                    pred_list = potential.get_ef_tensor(graph_list)
                loss_val, emae, fmae, smae = _loss(
                    target_list, pred_list, graph_list[Index.N_ATOMS]
                )
            grads = tape.gradient(loss_val, potential.model.trainable_variables)
            return loss_val, grads, pred_list, emae, fmae, smae

        for epoch in range(epochs):
            callback_list.on_epoch_begin(epoch=epoch, logs={"epoch": epoch})
            epoch_loss_avg = tf.keras.metrics.Mean()
            emae_avg = tf.keras.metrics.Mean()
            fmae_avg = tf.keras.metrics.Mean()
            smae_avg = tf.keras.metrics.Mean()
            for batch_index, batch in enumerate(mgb):
                callback_list.on_batch_begin(batch=batch_index)
                graph_batch, target_batch = batch
                lossval, grads, pred_list, emae, fmae, smae = train_one_step(
                    self.potential, graph_batch.as_tf().as_list(), target_batch
                )

                self.optimizer.apply_gradients(
                    zip(grads, self.potential.trainable_variables)
                )
                epoch_loss_avg.update_state(lossval)
                emae_avg.update_state(emae)
                fmae_avg.update_state(fmae)
                smae_avg.update_state(smae)
                logs = {
                    "loss": epoch_loss_avg.result().numpy(),
                    "MAE(E)": emae_avg.result().numpy(),
                    "MAE(F)": fmae_avg.result().numpy(),
                    "MAE(S)": smae_avg.result().numpy(),
                }
                callback_list.on_batch_end(batch=batch_index, logs=logs)

            epoch_logs = {
                "loss": epoch_loss_avg.result().numpy(),
                "MAE(E)": emae_avg.result().numpy(),
                "MAE(F)": fmae_avg.result().numpy(),
                "MAE(S)": smae_avg.result().numpy(),
            }

            epoch_loss_avg = tf.keras.metrics.Mean()
            emae_avg = tf.keras.metrics.Mean()
            fmae_avg = tf.keras.metrics.Mean()
            smae_avg = tf.keras.metrics.Mean()

            for batch_index, batch in enumerate(mgb_val):
                graph_batch, target_batch = batch
                lossval, emae, fmae, smae = _loss(
                    target_batch,
                    self.potential.get_efs_tensor(graph_batch.as_tf().as_list(), True),
                    graph_batch.n_atoms,
                )
                epoch_loss_avg.update_state(lossval)
                emae_avg.update_state(emae)
                fmae_avg.update_state(fmae)
                smae_avg.update_state(smae)

            epoch_logs.update(
                **{
                    "val_MAE": emae_avg.result().numpy()
                    + force_loss_ratio * fmae_avg.result().numpy()
                    + stress_loss_ratio * smae_avg.result().numpy(),
                    "val_MAE(E)": emae_avg.result().numpy(),
                    "val_MAE(F)": fmae_avg.result().numpy(),
                    "val_MAE(S)": smae_avg.result().numpy(),
                }
            )

            callback_list.on_epoch_end(epoch=epoch, logs=epoch_logs)
            mgb.on_epoch_end()

            if getattr(self.potential.model, "stop_training", False):
                break
