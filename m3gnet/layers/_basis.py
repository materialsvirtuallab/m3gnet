"""basis functions"""

import tensorflow as tf

from m3gnet.utils import Gaussian, SphericalBesselFunction

RBF_ALLOWED = {
    "Gaussian": {"params": ["centers", "width"], "class": Gaussian},
    "SphericalBessel": {
        "params": ["max_l", "max_n", "cutoff", "smooth"],
        "class": SphericalBesselFunction,
    },
}


class RadialBasisFunctions(tf.keras.layers.Layer):
    """
    Radial distribution function basis
    """

    def __init__(self, rbf_type: str = "Gaussian", **kwargs):
        """
        Args:
            rbf_type (str): RBF function type, choose between `Gaussian` or
                `SphericalBessel`
            **kwargs:
        """
        name = kwargs.pop("name", None)
        super().__init__(name=name)
        allowed_kwargs = {}
        keys = []
        for i, j in kwargs.items():
            if i in RBF_ALLOWED[rbf_type]["params"]:  # type: ignore
                allowed_kwargs.update({i: j})
                keys.append(i)

        missing_keys: set[str] = set(RBF_ALLOWED[rbf_type]["params"]) - set(keys)  # type: ignore
        if len(missing_keys) > 0:
            raise ValueError("kwargs ", missing_keys, " not present")
        self.allowed_kwargs = allowed_kwargs
        self.func = RBF_ALLOWED[rbf_type]["class"](**self.allowed_kwargs)  # type: ignore
        self.rbf_type = rbf_type

    def call(self, r: tf.Tensor, **kwargs):
        """
        Args:
            r (tf.Tensor): 1D radial distance tensor
            **kwargs:
        Returns: radial basis functions

        """
        return self.func(r)

    def get_config(self) -> dict:
        """
        Get config of the class for serialization
        Returns: dict

        """
        config = super().get_config()
        config.update({"rbf_type": self.rbf_type})
        config.update(self.allowed_kwargs)
        return config
