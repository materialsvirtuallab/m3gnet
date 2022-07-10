"""
Collate material graphs
"""
from typing import AnyStr, List, Optional, Tuple, Union, overload

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from ._types import Index, MaterialGraph


class MaterialGraphBatch(Sequence):
    """
    MaterialGraphBatch for generating batches of MaterialGraphs
    """

    def __init__(
        self,
        graphs: List[MaterialGraph],
        targets: Optional[np.ndarray] = None,
        batch_size: int = 128,
        shuffle: bool = True,
    ):
        """
        Args:
            graphs (list): list of MaterialGraph or MaskedMaterialGraph
            targets (np.ndarray): array of targets
            batch_size (int): batch size
            shuffle (bool): whether to shuffle graphs at the end of each epoch
        """
        self.graphs = graphs
        self.targets = np.array(targets, dtype="float32")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_n = len(graphs)
        self.max_step = int(np.ceil(self.total_n / batch_size))
        self.graph_index = np.arange(self.total_n)
        self._current_step = 0
        if self.shuffle:
            self.graph_index = np.random.permutation(self.graph_index)

    def __getitem__(self, index) -> Union[MaterialGraph, Tuple[MaterialGraph, np.ndarray]]:
        """
        Get the index-th batch of data. Returns a MaterialGraph object that
        contains many structures

        Args:
            index: int, the batch index

        Returns: MaterialGraph

        """
        graph_indices = self.graph_index[index * self.batch_size : (index + 1) * self.batch_size]
        graphs = [self.graphs[i] for i in graph_indices]
        new_graph: MaterialGraph = assemble_material_graph(graphs)
        targets: np.ndarray = None if self.targets is None else self.targets[graph_indices]
        if targets is None:
            return new_graph
        return new_graph, targets

    def __len__(self) -> int:
        return self.max_step

    def on_epoch_end(self) -> None:
        """
        Operations at the end of each training epoch
        """
        if self.shuffle:
            self.graph_index = np.random.permutation(self.graph_index)


class MaterialGraphBatchEnergyForceStress(MaterialGraphBatch):
    """
    MaterialGraphBatch for generating batches of MaterialGraphs
    """

    def __init__(
        self,
        graphs: List[MaterialGraph],
        energies: Union[List, np.ndarray] = None,
        forces: List[np.ndarray] = None,
        stresses: List[np.ndarray] = None,
        batch_size: int = 128,
        shuffle: bool = True,
    ):
        """
        Args:
            graphs (list): list of MaterialGraph or MaskedMaterialGraph
            energies (np.ndarray): array of targets
            forces (List): list of force matrix
            stresses (List or np.ndarray): list of stress matrix
            batch_size (int): batch size
            shuffle (bool): whether to shuffle graphs at the end of each epoch
        """
        super().__init__(graphs=graphs, targets=energies, batch_size=batch_size, shuffle=shuffle)
        self.forces = forces
        self.stresses = np.array(stresses, dtype="float32") if stresses is not None else None

    def __getitem__(self, index):
        graph_indices = self.graph_index[index * self.batch_size : (index + 1) * self.batch_size]
        graphs, energies = super().__getitem__(index)
        forces = np.concatenate([self.forces[i] for i in graph_indices], axis=0)
        forces = np.array(forces, dtype="float32")
        return_values = [graphs, (energies, forces)]
        if self.stresses is not None:
            stresses = np.array([self.stresses[i] for i in graph_indices])
            return_values[1] += (stresses,)
        return return_values


def _check_none_field(graph_list, field) -> bool:
    if any(getattr(g, field, None) is None for g in graph_list):
        return True
    return False


def _concatenate(list_of_arrays: List, name: AnyStr) -> Optional[np.ndarray]:
    """
    Concatenate list of array on the first dimension
    Args:
        list_of_arrays (List): list of arrays
        name (String): Used for error message

    Returns: concatenated array
    """
    num_none = sum(i is None for i in list_of_arrays)

    if num_none == len(list_of_arrays):
        return None

    if num_none == 0:
        if isinstance(list_of_arrays[0], tf.Tensor):
            return tf.concat(list_of_arrays, axis=0)
        return np.concatenate(list_of_arrays, axis=0)

    none_indices = [i for i, j in enumerate(list_of_arrays) if j is None]
    raise ValueError(
        f"The {name!r} properties of the graph indices {str(none_indices)!r} "
        f"are None while the rest graphs are not None. Hence the graphs "
        f"cannot be assembled."
    )


@overload
def assemble_material_graph(graphs: List[MaterialGraph]) -> MaterialGraph:
    ...


@overload
def assemble_material_graph(graphs: List[List]) -> List:
    ...


def assemble_material_graph(graphs):
    """
    Collate a list of MaterialGraph and form a single MaterialGraph

    Args:
        graphs: list of MaterialGraph

    Returns: a single MaterialGraph
    """
    if isinstance(graphs[0], List):
        return _assemble_material_graph_list(graphs)
    atoms = _concatenate([i.atoms for i in graphs], "atoms")
    bonds = _concatenate([i.bonds for i in graphs], "bonds")
    states = _concatenate([i.states for i in graphs], "states")
    bond_atom_indices = _concatenate([i.bond_atom_indices for i in graphs], "bond_atom_indices")

    n_atoms = _concatenate([i.n_atoms for i in graphs], "n_atoms")
    n_bonds = _concatenate([i.n_bonds for i in graphs], "n_bonds")

    n_atom_cumsum = np.cumsum([0] + [i.n_atom for i in graphs[:-1]])
    n_bond_every = [i.n_bond for i in graphs]
    bond_atom_indices += np.repeat(n_atom_cumsum, n_bond_every)[:, None]

    atom_positions = _concatenate([i.atom_positions for i in graphs], "atom_positions")
    bond_weights = _concatenate([i.bond_weights for i in graphs], "bond_weights")
    pbc_offsets = _concatenate([i.pbc_offsets for i in graphs], "pbc_offsets")
    lattices = _concatenate([i.lattices for i in graphs], "lattices")
    if graphs[0].has_threebody:
        triple_bond_indices = _concatenate([i.triple_bond_indices for i in graphs], "triple_bond_indices")
        n_bond_cumsum = np.cumsum([0] + n_bond_every[:-1])
        n_triple_s = _concatenate([i.n_triple_s for i in graphs], "n_triple_s")
        n_triple_s_temp = np.array([sum(i.n_triple_s) for i in graphs])
        triple_bond_indices += np.repeat(n_bond_cumsum, n_triple_s_temp)[:, None]
        triple_bond_lengths = _concatenate([i.triple_bond_lengths for i in graphs], "triple_bond_lengths")
        theta = _concatenate([i.theta for i in graphs], "theta")
        phi = _concatenate([i.phi for i in graphs], "phi")
        n_triple_ij = _concatenate([i.n_triple_ij for i in graphs], "n_triple_ij")
        n_triple_i = _concatenate([i.n_triple_i for i in graphs], "n_triple_i")

    else:
        triple_bond_indices = None
        triple_bond_lengths = None
        theta = None
        phi = None
        n_triple_ij = None
        n_triple_i = None
        n_triple_s = None

    return MaterialGraph(
        atoms=atoms,
        bonds=bonds,
        states=states,
        bond_atom_indices=bond_atom_indices,
        n_atoms=n_atoms,
        n_bonds=n_bonds,
        atom_positions=atom_positions,
        bond_weights=bond_weights,
        pbc_offsets=pbc_offsets,
        lattices=lattices,
        triple_bond_indices=triple_bond_indices,
        triple_bond_lengths=triple_bond_lengths,
        theta=theta,
        phi=phi,
        n_triple_ij=n_triple_ij,
        n_triple_i=n_triple_i,
        n_triple_s=n_triple_s,
    )


def _assemble_material_graph_list(graphs: List[List]) -> List:
    """
    Collate a list of MaterialGraph and form a single MaterialGraph

    Args:
        graphs: list of MaterialGraph

    Returns: a single MaterialGraph
    """

    graph = [None] * len(graphs[0])

    for i in [
        "atoms",
        "bonds",
        "states",
        "bond_atom_indices",
        "n_atoms",
        "n_bonds",
        "atom_positions",
        "bond_weights",
        "pbc_offsets",
        "lattices",
    ]:
        ind = getattr(Index, i.upper())
        graph[ind] = _concatenate([g[ind] for g in graphs], i)
    n_atoms = np.concatenate([i[Index.N_ATOMS] for i in graphs[:-1]])
    n_atom_cumsum = np.cumsum(np.concatenate([[0], n_atoms]))
    n_bond_every = [i[Index.BONDS].shape[0] for i in graphs]
    graph[Index.BOND_ATOM_INDICES] += np.repeat(n_atom_cumsum, n_bond_every)[:, None]

    if graphs[0][Index.N_TRIPLE_IJ] is not None:
        for i in [
            "triple_bond_indices",
            "n_triple_s",
            "triple_bond_lengths",
            "theta",
            "phi",
            "n_triple_ij",
            "n_triple_i",
        ]:
            ind = getattr(Index, i.upper())
            graph[ind] = _concatenate([g[ind] for g in graphs], i)
        n_bond_cumsum = np.cumsum([0] + n_bond_every[:-1])
        n_triple = np.array([sum(graph[Index.N_TRIPLE_S]) for graph in graphs])
        graph[Index.TRIPLE_BOND_INDICES] += np.repeat(n_bond_cumsum, n_triple)[:, None]
    return graph
