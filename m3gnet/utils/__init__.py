# -*- coding: utf-8 -*-
"""
Utilities for m3gnet
"""

from ._tf import get_length, get_segment_indices_from_n, \
    get_range_indices_from_n, repeat_with_n, \
    unsorted_segment_softmax, unsorted_segment_fraction, \
    broadcast_states_to_bonds, broadcast_states_to_atoms, register, \
    register_plain

from ._general import check_array_equal, check_shape_consistency, reshape_array
from ._math import get_spherical_bessel_roots, spherical_bessel_roots, \
    SphericalBesselFunction, SphericalHarmonicsFunction, combine_sbf_shf, \
    spherical_bessel_smooth

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
    "SphericalBesselFunction",
    "SphericalHarmonicsFunction",
    "combine_sbf_shf",
    "spherical_bessel_smooth"
]
