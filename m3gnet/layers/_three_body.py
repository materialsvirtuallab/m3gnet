"""
Three body basis expansion
"""
from typing import List

import tensorflow as tf

from m3gnet.graph import Index
from m3gnet.utils import (
    SphericalBesselFunction,
    SphericalHarmonicsFunction,
    combine_sbf_shf,
    register,
)


@register
class SphericalBesselWithHarmonics(tf.keras.layers.Layer):
    """
    Spherical bessel function as radial function and spherical harmonics as
    the angular function
    """

    def __init__(
        self, max_n: int, max_l: int, cutoff: float = 5.0, use_phi: bool = False, smooth: bool = False, **kwargs
    ):
        """

        Args:
            max_n (int): maximum number for radial expansion
            max_l (int): maximum number of angular expansion
            cutoff (float): cutoff radius
            use_phi (bool): whether to use phi. So far, we do not have the phi
                angle yet.
            smooth (bool): whether to use the smooth version of the radial
                function
            **kwargs:
        """
        super().__init__(**kwargs)
        self.sbf = SphericalBesselFunction(max_l=max_l, max_n=max_n, cutoff=cutoff, smooth=smooth)
        self.shf = SphericalHarmonicsFunction(max_l=max_l, use_phi=use_phi)
        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff
        self.use_phi = use_phi

    def call(self, graph: List, **kwargs) -> tf.Tensor:  # noqa
        """
        Args:
            graph (list): the list representation of a graph
            **kwargs:
        Returns: combined radial and spherical harmonic expansion of the
            distance and angle
        """
        sbf = self.sbf(graph[Index.TRIPLE_BOND_LENGTHS])
        shf = self.shf(graph[Index.THETA], graph[Index.PHI])
        return combine_sbf_shf(sbf, shf, max_n=self.max_n, max_l=self.max_l, use_phi=self.use_phi)

    def get_config(self) -> dict:
        """
        Get the config dict for serialization
        Returns: config dict
        """
        config = super().get_config()
        config.update(
            {
                "max_l": self.max_l,
                "max_n": self.max_n,
                "cutoff": self.cutoff,
                "use_phi": self.use_phi,
            }
        )
        return config
