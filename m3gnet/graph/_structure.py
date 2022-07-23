"""
Structure related calculations and etc.
"""

from typing import Tuple

import numpy as np
from ase import Atoms
from pymatgen.core.structure import Molecule, Structure
from pymatgen.optimization.neighbors import find_points_in_spheres

from m3gnet.config import DataType
from m3gnet.type import StructureOrMolecule


def get_fixed_radius_bonding(
    structure: StructureOrMolecule, cutoff: float = 5.0, numerical_tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get graph representations from structure within cutoff
    Args:
        structure (pymatgen Structure or molecule)
        cutoff (float): cutoff radius
        numerical_tol (float): numerical tolerance

    Returns:
        center_indices, neighbor_indices, images, distances
    """
    if isinstance(structure, Structure):
        lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
        cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    elif isinstance(structure, Molecule):
        lattice_matrix = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]], dtype=float)
        pbc = np.array([0, 0, 0], dtype=int)
        cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)

    elif isinstance(structure, Atoms):
        pbc = np.array(structure.pbc, dtype=int)
        if np.all(pbc < 0.1):
            lattice_matrix = np.array(
                [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]],
                dtype=float,
            )
        else:
            lattice_matrix = np.ascontiguousarray(structure.cell[:], dtype=float)

        cart_coords = np.ascontiguousarray(np.array(structure.positions), dtype=float)

    else:
        raise ValueError("structure type not supported")
    r = float(cutoff)

    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords,
        cart_coords,
        r=r,
        pbc=pbc,
        lattice=lattice_matrix,
        tol=numerical_tol,
    )
    center_indices = center_indices.astype(DataType.np_int)
    neighbor_indices = neighbor_indices.astype(DataType.np_int)
    images = images.astype(DataType.np_int)
    distances = distances.astype(DataType.np_float)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    return (
        center_indices[exclude_self],
        neighbor_indices[exclude_self],
        images[exclude_self],
        distances[exclude_self],
    )
