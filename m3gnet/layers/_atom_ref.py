"""
atomic reference. Used for predicting extensive properties.
"""

import logging
from typing import List, Optional

import numpy as np
import tensorflow as tf
from ase import Atoms
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Molecule, Structure

from m3gnet.config import DataType
from m3gnet.graph import Index
from m3gnet.utils import get_segment_indices_from_n, register

logger = logging.getLogger(__name__)


@register
class BaseAtomRef(tf.keras.layers.Layer):
    """
    Base AtomRef that predicts 0 correction
    """

    def call(self, graph: List, **kwargs):
        """

        Args:
            graph (list): list repr of a graph
            **kwargs:

        Returns: 0
        """
        return 0.0


@register
class AtomRef(BaseAtomRef):
    """
    Atom reference values. For example, if the average H energy is -20.0, and
    the average O energy is -10.0, the AtomRef predicts -20 * 2 + (-10.) =
    -50.0 for the atom reference energy for H2O
    """

    def __init__(
        self,
        property_per_element: Optional[np.ndarray] = None,
        max_z: int = 94,
        **kwargs,
    ):
        """
        Args:
            property_per_element (np.ndarray): element reference value
            max_z (int): maximum atomic number
            **kwargs:
        """
        super().__init__(**kwargs)
        if property_per_element is None:
            self.property_per_element = np.zeros(shape=(max_z + 1,))
        else:
            self.property_per_element = np.array(property_per_element).ravel()
        self.max_z = max_z

    def _get_feature_matrix(self, structs_or_graphs):
        n = len(structs_or_graphs)
        features = np.zeros(shape=(n, self.max_z + 1))
        for i, s in enumerate(structs_or_graphs):
            if isinstance(s, (Structure, Molecule)):
                atomic_numbers = [i.specie.Z for i in s.sites]
            elif isinstance(s, (list, np.ndarray)):
                atomic_numbers = s
            elif isinstance(s, Atoms):
                atomic_numbers = s.get_atomic_numbers()
            else:
                atomic_numbers = s.atoms[:, 0]
            features[i] = np.bincount(atomic_numbers, minlength=self.max_z + 1)
        return features

    def fit(self, structs_or_graphs, properties):
        """
        Fit the elemental reference values for the properties
        Args:
            structs_or_graphs (list): list of graphs or structures
            properties (np.ndarray): array of extensive properties
        Returns:
        """
        features = self._get_feature_matrix(structs_or_graphs)
        self.property_per_element = np.linalg.pinv(features.T.dot(features)).dot(features.T.dot(properties))
        string_prop = ""
        for i, j in enumerate(self.property_per_element):
            if abs(j) > 1e-5:
                string_prop += f"{str(Element.from_Z(i))}: {j:.5f}"
        logger.info("Property offset values: " + string_prop)
        return True

    def transform(self, structs_or_graphs, properties):
        """
        Correct the extensive properties by subtracting the atom reference
        values
        Args:
            structs_or_graphs (list): list of graphs or structures
            properties (np.ndarray): array of extensive properties
        Returns: corrected property values

        """
        properties = np.array(properties)
        atom_properties = self.predict_properties(structs_or_graphs)
        return properties - np.reshape(atom_properties, properties.shape)

    def inverse_transform(self, structs_or_graphs, properties):
        """
        Take the transformed values and get the original values
        Args:
            structs_or_graphs (list): list of graphs or structures
            properties (np.ndarray): array of extensive properties
        Returns: corrected property values

        """
        properties = np.array(properties)
        property_atoms = self.predict_properties(structs_or_graphs)
        final_properties = properties + np.reshape(property_atoms, properties.shape)
        return final_properties

    def predict_properties(self, structs_or_graphs):
        """
        Args:
            structs_or_graphs (list): calculate the atom summed property
                values
        Returns:

        """
        if not isinstance(structs_or_graphs, list):
            structs_or_graphs = [structs_or_graphs]
        features = self._get_feature_matrix(structs_or_graphs)
        return features.dot(self.property_per_element)

    def call(self, graph: List, **kwargs):
        """
        Args:
            graph (list): a list repr of a graph
            **kwargs:
        Returns:
        """
        atomic_numbers = graph[Index.ATOMS][:, 0]
        atom_energies = tf.gather(tf.cast(self.property_per_element, DataType.tf_float), atomic_numbers)
        res = tf.math.segment_sum(atom_energies, get_segment_indices_from_n(graph[Index.N_ATOMS]))
        return tf.reshape(res, (-1, 1))

    def set_property_per_element(self, property_per_element):
        """
        Set the property per atom value
        Args:
            property_per_element (np.ndarray): array of elemental properties,
                the i-th row is the elemental value for atomic number i.
        Returns:
        """
        self.property_per_element = property_per_element

    def get_config(self):
        """
        Get dict config for serialization
        Returns (dict):
        """
        config = super().get_config()
        config.update(**{"property_per_element": self.property_per_element, "max_z": self.max_z})
        return config
