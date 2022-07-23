"""
Classes to convert a structure into a graph
"""
import logging
from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from ase import Atoms
from pymatgen.core.structure import Structure

from m3gnet.config import DataType
from m3gnet.type import StructureOrMolecule
from m3gnet.utils import register, reshape_array

from ._batch import assemble_material_graph
from ._compute import include_threebody_indices
from ._structure import get_fixed_radius_bonding
from ._types import MaterialGraph

logger = logging.getLogger(__name__)


@register
class BaseGraphConverter(tf.keras.layers.Layer):
    """
    Basic graph converter that uses the atomic number as the node feature
    """

    def __init__(self, default_states=None, **kwargs):
        """
        Args:
            default_states (np.ndarray or tf.Tensor): the default state
                attributes for a MaterialGraph
            **kwargs:
        """

        self.default_states = default_states
        super().__init__(**kwargs)

    @staticmethod
    def get_atom_features(structure) -> np.ndarray:
        """
        Get atom features from structure, may be overwritten
        Args:
            structure: (Pymatgen.Structure) pymatgen structure
        Returns:
            List of atomic numbers
        """
        if isinstance(structure, Atoms):
            return np.array(structure.get_atomic_numbers(), dtype=DataType.np_int)
        return np.array([i.specie.Z for i in structure], dtype=DataType.np_int)

    def set_default_states(self, states=None):
        """
        set the default states for MaterialGraph
        Args:
            states (np.ndarray or tf.Tensor): the state attributes
        Returns:

        """
        self.default_states = states

    def get_states(self, structure: StructureOrMolecule):
        """
        Get the state attributes from a structure
        Args:
            structure (StructureOrMolecule): structure to compute the states

        Returns: tf.Tensor or np.ndarray

        """
        states = getattr(structure, "states", None)
        if states is None:
            states = self.default_states
        return states

    @abstractmethod
    def convert(self, structure: StructureOrMolecule, **kwargs) -> MaterialGraph:
        """
        Convert the structure into a graph
        Args:
            structure:
            **kwargs:
        Returns:

        """

    def convert_many(self, structures: List[StructureOrMolecule], **kwargs) -> MaterialGraph:
        """
        Convert many structures into one single graph
        Args:
            structures: List of structures
            **kwargs:
        Returns: MaterialGraph
        """
        graphs = [self.convert(structure, **kwargs) for structure in structures]
        return assemble_material_graph(graphs)

    def __call__(self, structure: StructureOrMolecule, *args, **kwargs) -> MaterialGraph:
        """
        A thin wrapper for calling `convert` method
        Args:
            structure:
            *args:
            **kwargs:
        Returns:

        """
        return self.convert(structure)


@register
class RadiusCutoffGraphConverter(BaseGraphConverter):
    """
    Constructing a material graph with fixed radius cutoff
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        has_threebody: bool = True,
        threebody_cutoff: Optional[float] = None,
        **kwargs,
    ):
        """

        Args:
            cutoff: float, cutoff radius
            atom_converter: Converter, atom converter
            bond_converter: Converter, bond converter
            three_body_cutoff: float, three-body cutoff radius
            **kwargs:
        """
        self.cutoff = cutoff
        self.has_threebody = has_threebody

        if has_threebody:
            if threebody_cutoff is None:
                threebody_cutoff = cutoff - 1.0
            if threebody_cutoff > cutoff:
                raise ValueError("Three body cutoff has to be smaller than two " "body")
        self.threebody_cutoff = threebody_cutoff

        super().__init__(**kwargs)

    def convert(self, structure: StructureOrMolecule, **kwargs) -> MaterialGraph:
        """
        Convert the structure into graph
        Args:
            structure: Structure or Molecule
        Returns
            MaterialGraph
        """

        if isinstance(structure, Atoms):
            atom_positions = np.asarray(structure.get_positions(), dtype=DataType.np_float)
        else:
            atom_positions = np.array(structure.cart_coords, dtype=DataType.np_float)
        state_attributes = self.get_states(structure)

        sender_indices, receiver_indices, images, distances = get_fixed_radius_bonding(structure, self.cutoff)

        if np.size(np.unique(sender_indices)) < len(structure):
            logger.warning("Isolated atoms found in the structure")

        bonds = distances[:, None]
        bond_atom_indices = np.array([sender_indices, receiver_indices], dtype=DataType.np_int).T
        pbc_offsets = images.astype(DataType.np_int)

        mg = MaterialGraph(  # type: ignore
            bonds=bonds,
            bond_weights=distances,
            bond_atom_indices=bond_atom_indices,  # noqa
            pbc_offsets=pbc_offsets,
        )  # noqa

        n_atom = len(structure)

        atoms = reshape_array(self.get_atom_features(structure), [n_atom, None])
        if state_attributes is not None:
            state_attributes = reshape_array(state_attributes, [1, None])

        n_bonds = np.array([mg.bonds.shape[0]], dtype=DataType.np_int)  # type: ignore
        mg = mg.replace(
            states=state_attributes,
            atoms=atoms,
            atom_positions=atom_positions,
            n_atoms=np.array([n_atom], dtype=DataType.np_int),
            n_bonds=n_bonds,
        )

        if isinstance(structure, Structure):
            mg = mg.replace(lattices=np.array(structure.lattice.matrix.reshape((1, 3, 3)), dtype=DataType.np_float))

        if isinstance(structure, Atoms):
            mg = mg.replace(lattices=np.array(structure.cell[:].reshape((1, 3, 3)), dtype=DataType.np_float))

        if self.has_threebody:
            mg = include_threebody_indices(mg, self.threebody_cutoff)
        return mg

    def __str__(self):
        s = f"<RadiusCutoffGraphConverter cutoff={self.cutoff}" f" threebody_cutoff={self.threebody_cutoff} >"
        return s

    def __repr__(self):
        return str(self)

    def get_config(self) -> Dict:
        """
        Get serialized dictionary
        Returns: Dict

        """
        return {
            "cutoff": self.cutoff,
            "threebody_cutoff": self.threebody_cutoff,
            "name": self.name,
            "trainable": self.trainable,
            "dtype": self.dtype,
            "has_threebody": self.has_threebody,
        }
