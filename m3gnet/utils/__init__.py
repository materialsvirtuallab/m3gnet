"""
Utilities for m3gnet
"""

from ._general import check_array_equal, check_shape_consistency, reshape_array
from ._math import (
    Gaussian,
    SphericalBesselFunction,
    SphericalHarmonicsFunction,
    combine_sbf_shf,
    get_spherical_bessel_roots,
    spherical_bessel_roots,
    spherical_bessel_smooth,
)
from ._tf import (
    broadcast_states_to_atoms,
    broadcast_states_to_bonds,
    get_length,
    get_range_indices_from_n,
    get_segment_indices_from_n,
    register,
    register_plain,
    repeat_with_n,
    unsorted_segment_fraction,
    unsorted_segment_softmax,
)

__all__ = [
    "get_length",
    "get_segment_indices_from_n",
    "get_range_indices_from_n",
    "repeat_with_n",
    "unsorted_segment_softmax",
    "unsorted_segment_fraction",
    "broadcast_states_to_atoms",
    "broadcast_states_to_bonds",
    "register",
    "register_plain",
    "check_array_equal",
    "check_shape_consistency",
    "reshape_array",
    "get_spherical_bessel_roots",
    "spherical_bessel_roots",
    "Gaussian",
    "SphericalBesselFunction",
    "SphericalHarmonicsFunction",
    "combine_sbf_shf",
    "spherical_bessel_smooth",
]
