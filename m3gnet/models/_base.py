"""
Base model
"""

import math
import platform
from abc import ABC
from typing import List, Union

import tensorflow as tf
from pymatgen.core.structure import Molecule, Structure

from m3gnet.config import DataType
from m3gnet.graph import Index, MaterialGraph, assemble_material_graph
from m3gnet.type import StructureOrMolecule
from m3gnet.utils import register, repeat_with_n

PLATFORM = platform.platform()


@register
class GraphModelMixin(tf.keras.layers.Layer):
    """
    GraphModelMixin adds the following functionality to a graph model
        - predict_structure
        - predict_structures
        - predict_graph
        - predict_graphs
    """

    def predict_structure(self, structure: StructureOrMolecule) -> tf.Tensor:
        """
        predict properties from structure
        Args:
            structure (StructureOrMolecule): a pymatgen Structure/Molecule,
                or ase.Atoms
        Returns: predicted values
        """
        return self.predict_graph(self.graph_converter(structure))

    def predict_structures(self, structures: List[StructureOrMolecule], batch_size: int = 128) -> tf.Tensor:
        """
        predict properties from structures
        Args:
            structures (list[StructureOrMolecule]): a list of structures
            batch_size (int): batch size for the prediction
        Returns: predicted values
        """
        graph_list = [self.graph_converter(i) for i in structures]
        return self.predict_graphs(graph_list, batch_size)

    def predict_graph(self, graph: Union[MaterialGraph, List]) -> tf.Tensor:
        """
        predict properties from a graph
        Args:
            graph (Union[MaterialGraph, List]): a material graph, either in
                object or list repr
        Returns: predicted property
        """
        if isinstance(graph, MaterialGraph):
            graph = graph.as_list()
        return self.call(graph)

    def predict_graphs(self, graph_list: List[Union[MaterialGraph, List]], batch_size: int = 128) -> tf.Tensor:
        """
        predict properties from graphs
        Args:
            graph_list (List[Union[MaterialGraph, List]]): a list of material
                graph, either in object or list repr
            batch_size (int): batch size
        Returns: predicted properties
        """
        n = len(graph_list)
        use_graph = bool(isinstance(graph_list[0], MaterialGraph))
        n_steps = math.ceil(n / batch_size)
        predicted = []
        for i in range(n_steps):
            graphs = graph_list[batch_size * i : batch_size * (i + 1)]
            graph = assemble_material_graph(graphs)  # type: ignore
            if use_graph:
                results = self.call(graph.as_list())
            else:
                results = self.call(graph)
            predicted.append(results)
        return tf.concat(predicted, axis=0)


@register
class BasePotential(tf.keras.Model, ABC):
    """
    Potential abstract class
    """

    def get_energies(self, graph: List):
        """
        Compute the energy of a MaterialGraph
        Args:
            graph: List, a graph from structure
        Returns: energy values, size [Ns]
        """
        return 0.0

    def get_forces(self, graph: List):
        """
        Compute forces of a graph given the atom positions
        Args:
            graph: List, a graph from structure

        Returns: forces [Na, 3]
        """
        return self.get_efs(graph)[1]

    def get_stresses(self, graph: List):
        """
        Compute stress of a graph given the atom positions
        Args:
            graph: List, a graph from structure
        Returns: stresses [Ns, 3, 3]
        """
        return self.get_efs(graph)[2]

    def get_ef(self, obj: Union[StructureOrMolecule, MaterialGraph, List]) -> tuple:
        """
        get energy and force from a Structure, a graph or a list repr of a
        graph
        Args:
            obj (Union[StructureOrMolecule, MaterialGraph, List]): a structure,
                material graph or list repr of a graph
        Returns:
        """
        return self.get_efs(obj, include_stresses=False)

    @tf.function(experimental_relax_shapes=True)
    def get_ef_tensor(self, graph: List[tf.Tensor]) -> tuple:
        """
        get energy and force from a list repr of graph
        Args:
            graph (List[tf.Tensor]: a list repr of a graph
        Returns:
        """
        return self.get_efs_tensor(graph, include_stresses=False)

    def get_efs(
        self,
        obj: Union[StructureOrMolecule, MaterialGraph, List],
        include_stresses: bool = True,
    ):
        """
        get energy and force from a Structure, a graph or a list repr of a
        graph
        Args:
            obj (Union[StructureOrMolecule, MaterialGraph, List]): a structure,
                material graph or list repr of a graph
            include_stresses (bool): whether to include stress
        Returns:
        """
        if isinstance(obj, Structure):
            obj = self.model.graph_converter(obj)
        if isinstance(obj, MaterialGraph):
            obj = obj.as_tf().as_list()
        return self.get_efs_tensor(obj, include_stresses=include_stresses)

    @tf.function(experimental_relax_shapes=True)
    def get_efs_tensor(self, graph: List[tf.Tensor], include_stresses: bool = True) -> tuple:
        """
        get energy and force from a list repr of a
        graph
        Args:
            graph (List[tf.Tensor]): a list repr of a graph
            include_stresses (bool): whether to include stress
        Returns:
        """
        with tf.GradientTape() as tape:
            tape.watch(graph[Index.ATOM_POSITIONS])
            if include_stresses:
                graph = graph[:]
                strain = tf.zeros_like(graph[Index.LATTICES])
                tape.watch(strain)
                graph[Index.LATTICES] = tf.matmul(graph[Index.LATTICES], (tf.eye(3)[None, ...] + strain))

                strain_augment = repeat_with_n(strain, graph[Index.N_ATOMS])
                graph[Index.ATOM_POSITIONS] = tf.keras.backend.batch_dot(
                    graph[Index.ATOM_POSITIONS], (tf.eye(3)[None, ...] + strain_augment)
                )
                volume = tf.linalg.det(graph[Index.LATTICES])
            energies = self.get_energies(graph)
            derivatives = {"forces": graph[Index.ATOM_POSITIONS]}
            if include_stresses:
                derivatives["stresses"] = strain  # type: ignore
            if "macOS" in PLATFORM and "arm64" in PLATFORM and tf.config.list_physical_devices("GPU"):
                # This is a workaround for a bug in tensorflow-metal that fails when tape.gradient is called.
                with tf.device("/cpu:0"):
                    derivatives = tape.gradient(energies, derivatives)
            else:
                derivatives = tape.gradient(energies, derivatives)

        forces = -derivatives["forces"]
        forces = tf.cast(tf.convert_to_tensor(forces), DataType.tf_float)
        results: tuple = (energies, forces)
        # eV/A^3 to GPa
        if include_stresses:
            stresses = 1 / volume[:, None, None] * derivatives["stresses"] * 160.21766208
            stresses = tf.cast(tf.convert_to_tensor(stresses), DataType.tf_float)
            results += (stresses,)
        return results

    def call(
        self,
        graph: Union[MaterialGraph, Structure, Molecule, List],
        include_forces: bool = True,
        include_stresses: bool = True,
    ):
        """
        Apply the potential to a graph
        Args:
            graph (Union[MaterialGraph, Structure, Molecule, List]): a
                structure, molecule, graph or a list repr of a graph
            include_forces (bool): whether to include forces as outputs
            include_stresses (bool: whether to include stresses as outputs
        Returns: energy [forces, stress]
        """
        if isinstance(graph, (Structure, Molecule)):
            graph = self.graph_converter.convert(graph)
        efs = self.get_efs(graph, include_stresses=include_stresses)
        results: tuple = (efs[0],)
        if include_forces:
            results += (efs[1],)
        if include_stresses:
            results += (efs[2],)
        if len(results) == 1:
            return results[0]
        return results


@register
class Potential(BasePotential):
    """
    Defines the Potential class. New potential should subclass from this class
    and define the "get_energies" method
    """

    def __init__(self, model: tf.keras.layers.Layer, **kwargs):
        """
        Args:
            model (tf.keras.layers.Layer): a callable model that predict values
                from a graph
            **kwargs:
        """
        super().__init__(**kwargs)
        self.model = model
        self.graph_converter = model.graph_converter

    @tf.function(experimental_relax_shapes=True)
    def get_energies(self, graph: List) -> tf.Tensor:
        """
        get energies from a list repr of a graph
        Args:
            graph (List): list repr of a graph
        Returns:
        """
        return self.model(graph)
