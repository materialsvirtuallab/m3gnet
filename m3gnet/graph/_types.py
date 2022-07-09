"""
Graph types.
"""

from dataclasses import dataclass, replace
from typing import ClassVar, List, Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from m3gnet.utils import check_array_equal, check_shape_consistency


def _to_numpy(x):
    if isinstance(x, Sequence):
        return np.array(x)
    if isinstance(x, tf.Tensor):
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x
    raise ValueError("Cannot convert to numpy")


#  graph features
ATOMS = "atoms"
BONDS = "bonds"
STATES = "states"
GRAPH_FEATURES = (ATOMS, BONDS, STATES)

# atom positions, some pretrained may need explicit
# atom positions
ATOM_POSITIONS = "atom_positions"
BOND_WEIGHTS = "bond_weights"

# atom indices that define bonds
BOND_ATOM_INDICES = "bond_atom_indices"

# number of atoms and graphs in each structure
N_ATOMS = "n_atoms"
N_BONDS = "n_bonds"
ATOM_BOND_NUMBERS = (N_ATOMS, N_BONDS)

# periodic boundary fields
PBC_OFFSETS = "pbc_offsets"
LATTICES = "lattices"
PBC_FIELDS = (PBC_OFFSETS, LATTICES)

# threebody information
TRIPLE_BOND_INDICES = "triple_bond_indices"
TRIPLE_BOND_LENGTHS = "triple_bond_lengths"
THETA = "theta"
PHI = "phi"
N_TRIPLE_IJ = "n_triple_ij"
N_TRIPLE_I = "n_triple_i"
N_TRIPLE_S = "n_triple_s"


ALL_FIELDS = (
    ATOMS,
    BONDS,
    STATES,
    ATOM_POSITIONS,
    BOND_ATOM_INDICES,
    PBC_OFFSETS,
    N_ATOMS,
    N_BONDS,
    BOND_WEIGHTS,
    LATTICES,
    TRIPLE_BOND_INDICES,
    TRIPLE_BOND_LENGTHS,
    THETA,
    PHI,
    N_TRIPLE_IJ,
    N_TRIPLE_I,
    N_TRIPLE_S,
)


GRAPH_INDEX = {i: j for j, i in enumerate(ALL_FIELDS)}


class Index:
    """
    Get integer indices for each field in MaterialGraph
    """

    ATOMS = 0
    BONDS = 1
    STATES = 2
    ATOM_POSITIONS = 3
    BOND_ATOM_INDICES = 4
    PBC_OFFSETS = 5
    N_ATOMS = 6
    N_BONDS = 7
    BOND_WEIGHTS = 8
    LATTICES = 9
    TRIPLE_BOND_INDICES = 10
    TRIPLE_BOND_LENGTHS = 11
    THETA = 12
    PHI = 13
    N_TRIPLE_IJ = 14
    N_TRIPLE_I = 15
    N_TRIPLE_S = 16


class AttributeUpdateMixin:
    """
    Mixin class for updating the fields of the graphs and checking
    graph consistency
    """

    def _check_graph(self) -> None:
        """
        Check if the graph is valid
        """

    def replace(self, **kwargs):
        """
        Replace a graph field
        Args:
            **kwargs: dictionary for replacements

        Returns:

        """
        return replace(self, **kwargs)  # noqa

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        for i, j in self.__dict__.items():
            is_equal = check_array_equal(j, other.__dict__[i])
            if not is_equal:
                print(
                    f"{i} is different, one is ",
                    j,
                    "and the other is",
                    other.__dict__[i],
                )
                return False
        return True

    def _check_shapes(self, shape_dict):
        checks = [check_shape_consistency(getattr(self, i), j) for i, j in shape_dict.items()]
        names = list(shape_dict.keys())
        error_msg = ""
        for check, name in zip(checks, names):
            if not check:
                error_msg += f"{name} shape is {getattr(self, name).shape}, "
                error_msg += f"while expecting {shape_dict[name]}"

        if len(error_msg) > 0:
            raise RuntimeError("The data shapes do not comply\n" + error_msg)


def _check_n(fields, name, dim=0):
    """
    Check the field consistency
    Args:
        fields (list): a list of fields
        name (str): name to use when raising error
        dim (int): dimension for comparing

    Returns: int or None
    """
    n_candidates = []
    for i in fields:
        if i is not None:
            n_candidates.append(i.shape[dim])
    if len(n_candidates) > 0:
        if len(list(set(n_candidates))) > 1:
            raise ValueError(f"{name} inconsistent")
        return n_candidates[0]
    return None


@dataclass(eq=False)
class MaterialGraph(AttributeUpdateMixin):
    """
    Material graph by an edge list graph representation.
    In this representation, no padding of atoms or bonds are needed.

    Full available data names are the following, assuming the
    number of atoms is `n_atom`, number of bonds is `n_bond` and
    number of structure is `n_struct`.

    The attributes are as follows

        - `atoms` (np.ndarray): atom attributes with shape [n_atom, None]
        - `bonds` (np.ndarray): bond attributes with shape [n_bond, None]
            states (np.ndarray): state attributes with shape [n_struct, None]
            bond_atom_indices (np.ndarray): int indices for pairs of atoms,
            shape [n_bond, 2]
        - `n_atoms` (np.ndarray): int array, number of atoms in each structure,
            [n_struct]
        - `n_bonds` (np.ndarray): int array, number of bonds in each structure,
            [n_struct]
        - `atom_positions` (np.ndarray): float type, atom position array, shape
            [n_atom, 3]
        - `bond_weights` (np.ndarray): float type, bond lengths, shape [n_bond]
        - `pbc_offsets` (np.ndarray): int type, periodic boundary offset
             vectors, shape [n_bond, 3].
        - `lattices` (np.ndarray): float type, lattice matrices for all
            structures, shape [n_struct, 3, 3]
        - `triple_bond_indices` (np.ndarray): int type, the bond indices for
            forming triple bonds
        - `triple_bond_lengths` (np.ndarray): float type, the triple bond
            lengths
        - `theta` (np.ndarray): float type,the azimuthal angle of the triple
            bond
        - `phi` (np.ndarray): float type, the polar angle of the triple bond
        - `n_triple_ij` (np.ndarray): int type, the number of triple bonds for
            each bond
        -  `n_triple_i` (np.ndarray): int type, the number of triple bonds for
            each atom
        - `n_triple_s` (np.ndarray): int type, the number of triple bonds
            for each structure
    """

    atoms: Optional[Union[np.ndarray, tf.Tensor]] = None  # [n_atom, None]
    bonds: Optional[Union[np.ndarray, tf.Tensor]] = None  # [n_bond, None]
    states: Optional[Union[np.ndarray, tf.Tensor]] = None  # [n_struct, None]
    # [n_bond, 2], int
    bond_atom_indices: Optional[Union[np.ndarray, tf.Tensor]] = None
    n_atoms: Optional[Union[np.ndarray, tf.Tensor]] = None  # [n_struct], int
    n_bonds: Optional[Union[np.ndarray, tf.Tensor]] = None  # [n_struct], int

    # the following are optional
    # [n_atom, 3], float
    atom_positions: Optional[Union[np.ndarray, tf.Tensor]] = None
    # [n_bond], float
    bond_weights: Optional[Union[np.ndarray, tf.Tensor]] = None
    # [n_bond, 3], int
    pbc_offsets: Optional[Union[np.ndarray, tf.Tensor]] = None
    # [n_struct, 3, 3], float
    lattices: Optional[Union[np.ndarray, tf.Tensor]] = None

    # The following are the three body information
    # [n_triple, 2], int
    triple_bond_indices: Optional[Union[np.ndarray, tf.Tensor]] = None

    # [n_triple, ], float
    triple_bond_lengths: Optional[Union[np.ndarray, tf.Tensor]] = None
    # [n_triple, ], float
    theta: Optional[Union[np.ndarray, tf.Tensor]] = None
    # [n_triple, ], float
    phi: Optional[Union[np.ndarray, tf.Tensor]] = None
    # [n_bond, ], int
    n_triple_ij: Optional[Union[np.ndarray, tf.Tensor]] = None
    # [n_atom, ], int
    n_triple_i: Optional[Union[np.ndarray, tf.Tensor]] = None
    # [n_struct, ], int
    n_triple_s: Optional[Union[np.ndarray, tf.Tensor]] = None

    # the following are class attributes
    base_attributes: ClassVar[List[str]] = [
        "atoms",
        "bonds",
        "bond_atom_indices",
        "n_atoms",
        "n_bonds",
    ]

    graph_with_position_attributes: ClassVar[List[str]] = [
        "atoms",
        "bond_atom_indices",
        "atom_positions",
        "bond_weights",
        "n_atoms",
        "n_bonds",
        "pbc_offsets",
        "lattices",
    ]

    three_body_attributes: ClassVar[List[str]] = [
        "triple_bond_indices",
        "triple_bond_lengths",
        "theta",
        "phi",
        "n_triple_ij",
        "n_triple_i",
        "n_triple_s",
    ]

    def __repr__(self):
        attributes = ALL_FIELDS
        shapes = []
        for attr in attributes:
            if getattr(self, attr) is None:
                continue
            shapes.append(attr + " " + str(getattr(self, attr).dtype)[9:-2] + f" {str(getattr(self, attr).shape)}")
        string = "\n" + "\n".join(shapes)
        return "<MaterialGraph with the following data shapes: " + string + ">"

    def as_tf(self) -> "MaterialGraph":
        """
        Convert each field to tensorflow tensors
        Returns: tf.Tensor
        """
        d = {i: _maybe_tensor(getattr(self, i)) for i in ALL_FIELDS}
        mg = MaterialGraph(**d)  # type: ignore
        return mg

    @property
    def has_threebody(self) -> bool:
        """
        Whether the graph has threebody indices
        Returns: boolean value indicator

        """
        if self.n_triple_ij is not None:
            return True
        return False

    @property
    def n_atom(self) -> Optional[int]:
        """
        number of atoms in the graph
        Returns: int or None
        """
        return None if self.atoms is None else self.atoms.shape[0]

    @property
    def n_bond(self) -> Optional[int]:
        """
        number of bonds
        Returns: int or None
        """
        return None if self.bonds is None else self.bonds.shape[0]

    @property
    def n_struct(self) -> Optional[int]:
        """
        number of structures
        Returns: int or None
        """
        if self.states is not None:
            return self.states.shape[0]
        if self.n_atoms is not None:
            return len(self.n_atoms)
        return None

    def as_list(self) -> List:
        """
        Convert the MaterialGraph to list representation
        Returns:

        """
        return [getattr(self, i) for i in ALL_FIELDS]

    @classmethod
    def from_list(cls, graph_list):
        """
        Construct the MaterialGraph object from a list representation
        Args:
            graph_list (list): list representation of a MaterialGraph

        Returns: MaterialGraph instance

        """
        graph_list = [np.array(i) if i is not None else i for i in graph_list]
        return cls(**dict(zip(ALL_FIELDS, graph_list)))

    def copy(self) -> "MaterialGraph":
        """
        Copy the MaterialGraph to a new one
        Returns: MaterialGraph
        """

        graph_list = self.as_list()
        graph_list_copy = [i.copy() if i is not None else None for i in graph_list]
        return MaterialGraph.from_list(graph_list_copy)


def _maybe_tensor(inp):
    """
    If the input is not a tensor, convert it to tensor
    Args:
        inp (object): input tensor or array or list

    Returns: tf.Tensor

    """
    if inp is not None:
        return tf.constant(inp)
    return None
