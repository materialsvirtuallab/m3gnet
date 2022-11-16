import numpy as np
import itertools


def compute_threebody(bond_atom_indices, n_atoms):
    """
    Calculate the three body indices from pair atom indices

    Args:
        bond_atom_indices (np.ndarray): pair atom indices
        n_atoms (list): number of atoms in each structure.

    Returns:
        triple_bond_indices (np.ndarray): bond indices that form three-body
        n_triple_ij (np.ndarray): number of three-body angles for each bond
        n_triple_i (np.ndarray): number of three-body angles each atom
        n_triple_s (np.ndarray): number of three-body angles for each structure
    """
    n_bond = len(bond_atom_indices)
    n_struct = len(n_atoms)
    n_atom = np.sum(n_atoms)

    n_bond_per_atom = [np.sum(bond_atom_indices[:, 0] == i) for i in range(n_atom)]

    n_triple_i = np.zeros(n_atom, dtype=np.int32)
    n_triple_ij = np.zeros(n_bond, dtype=np.int32)
    n_triple_s = np.zeros(n_struct, dtype=np.int32)
    triple_bond_indices = []

    n_triple = 0
    start = 0

    for i, bpa in enumerate(n_bond_per_atom):
        n_triple_temp = bpa * (bpa - 1)
        n_triple_ij[start : start + bpa] = bpa - 1
        n_triple += n_triple_temp
        n_triple_i[i] = n_triple_temp
        for j, k in itertools.permutations(range(bpa), 2):
            triple_bond_indices.append([start + j, start + k])
        start += bpa

    start = 0
    for i, n in enumerate(n_atoms):
        end = start + n
        for j in range(start, end):
            n_triple_s[i] += n_triple_i[j]
        start = end

    return triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s
