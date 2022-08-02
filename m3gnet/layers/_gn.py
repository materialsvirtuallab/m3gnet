"""
Materials Graph Network
"""
from copy import deepcopy
from typing import List, Optional

import tensorflow as tf
from m3gnet.utils import register

from ._atom import AtomNetwork
from ._base import GraphUpdate, GraphUpdateFunc
from ._bond import BondNetwork, PairRadialBasisExpansion
from ._core import Embedding
from ._state import StateNetwork


@register
class GraphNetworkLayer(GraphUpdate):
    """
    A graph network layer features bond/atom/state update in the sequence
    bond -> atom -> state. The input and output of each step are graphs/graph
    converted lists
    """

    def __init__(
        self,
        bond_network: Optional[GraphUpdate] = None,
        atom_network: Optional[GraphUpdate] = None,
        state_network: Optional[GraphUpdate] = None,
        **kwargs,
    ):
        """

        Args:
            bond_network (GraphUpdate): bond update network
            atom_network (GraphUpdate): atom update network
            state_network (GraphUpdate): state update network
            **kwargs:
        """
        self.bond_network = bond_network or GraphUpdate()
        self.atom_network = atom_network or GraphUpdate()
        self.state_network = state_network or GraphUpdate()
        super().__init__(**kwargs)

    def call(self, graph: List, **kwargs) -> List:
        """
        Args:
            graph (List): a graph in list representation
            **kwargs:
        Returns: tf.Tensor
        """
        out = self.state_network(self.atom_network(self.bond_network(graph)))
        return out

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns: dict
        """
        config = super().get_config()
        config.update(
            {
                "bond_network": self.bond_network,
                "atom_network": self.atom_network,
                "state_network": self.state_network,
            }
        )
        return config


def _get_bond_featurizer(nfeat_bond, n_bond_types, bond_embedding_dim, rbf_type, kwargs):
    # bond network settings for the featurization layer
    # no need for bond network
    if nfeat_bond is not None:
        return BondNetwork()

    # no feature length, meaning that the features are either
    # integer  types or just distances
    if bond_embedding_dim is not None:
        # bond attributes are integer types
        return GraphFieldEmbedding(
            nvocal=n_bond_types,
            embedding_dim=bond_embedding_dim,
            field="bonds",
            name="bond_embedding",
        )

    if rbf_type.lower() == "gaussian":
        centers = kwargs.pop("centers", None)
        width = kwargs.pop("width", None)
        if centers is None and width is None:
            raise ValueError(
                "If the bond attributes are single float values, "
                "we expect the value to be expanded before passing "
                "to the pretrained. Therefore, `centers` and `width` for "
                "Gaussian basis expansion are needed"
            )
        # bond attributes are distances
        bond_network = PairRadialBasisExpansion(rbf_type="Gaussian", centers=centers, width=width)
        return bond_network

    if rbf_type.lower() == "sphericalbessel":
        max_l = kwargs.pop("max_l")
        max_n = kwargs.pop("max_n")
        cutoff = kwargs.pop("cutoff")
        smooth = kwargs.pop("smooth", False)
        return PairRadialBasisExpansion(
            rbf_type="SphericalBessel",
            max_l=max_l,
            max_n=max_n,
            cutoff=cutoff,
            smooth=smooth,
        )
    raise ValueError("Cannot derive bond network type")


@register
class GraphFeaturizer(GraphNetworkLayer):
    """
    Graph featurizer that does several things to convert an initial graph
    with atomic number atom attributes and bond distance bond attributes to
    a graph with proper feature dimensions
    """

    def __init__(
        self,
        nfeat_bond: Optional[int] = None,
        nfeat_atom: Optional[int] = None,
        nfeat_state: Optional[int] = None,
        n_bond_types: Optional[int] = None,
        n_atom_types: Optional[int] = None,
        n_state_types: Optional[int] = None,
        bond_embedding_dim: Optional[int] = None,
        atom_embedding_dim: Optional[int] = None,
        state_embedding_dim: Optional[int] = None,
        rbf_type="SphericalBessel",
        **kwargs,
    ):
        """

        Args:
            nfeat_bond (int): bond feature dimension
            nfeat_atom (int): atom feature dimension
            nfeat_state (int): state feature dimension
            n_bond_types (int): number of bond types, used only when the
                bond is categorical
            n_atom_types (int): number of atom types
            n_state_types (int): number of state types, used only when state is
                of categorical type
            bond_embedding_dim (int): embedding dimension for bond
            atom_embedding_dim (int): embedding dimension for atom
            state_embedding_dim (int): embedding dimension for state
            rbf_type (str): radial basis function type, choose between
                "SphericalBessel" and "Gaussian"
            **kwargs:
        """
        self.kwargs = deepcopy(kwargs)

        # bond network settings for the featurization layer
        bond_network = _get_bond_featurizer(nfeat_bond, n_bond_types, bond_embedding_dim, rbf_type, kwargs)
        # atom network for the featurization layer
        if nfeat_atom is None:
            if n_atom_types is None or atom_embedding_dim is None:
                raise ValueError("Either specify nfeat_atom or " "n_atom_types and atom_embedding_dim")
            atom_network = GraphFieldEmbedding(
                nvocal=n_atom_types + 1,
                embedding_dim=atom_embedding_dim,
                field="atoms",
                name="atom_embedding",
            )
        else:
            atom_network = AtomNetwork()

        # state network for the featurization layer
        if nfeat_state is None and state_embedding_dim is not None:
            if n_state_types is None:
                raise ValueError("Either specify nfeat_state or " "n_state_types and state_embedding_dim")
            state_network = GraphFieldEmbedding(
                nvocal=n_state_types,
                embedding_dim=state_embedding_dim,
                field="states",
                name="state_embedding",
            )
        else:
            state_network = StateNetwork()

        self.nfeat_bond = nfeat_bond
        self.nfeat_atom = nfeat_atom
        self.nfeat_state = nfeat_state
        self.n_bond_types = n_bond_types
        self.n_atom_types = n_atom_types
        self.n_state_types = n_state_types
        self.bond_embedding_dim = bond_embedding_dim
        self.atom_embedding_dim = atom_embedding_dim
        self.state_embedding_dim = state_embedding_dim
        self.rbf_type = rbf_type
        super().__init__(bond_network=bond_network, atom_network=atom_network, state_network=state_network, **kwargs)

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns:
        """
        config = tf.keras.layers.Layer.get_config(self)
        config.update(
            nfeat_bond=self.nfeat_bond,
            nfeat_atom=self.nfeat_atom,
            nfeat_state=self.nfeat_state,
            n_bond_types=self.n_bond_types,
            n_atom_types=self.n_atom_types,
            n_state_types=self.n_state_types,
            bond_embedding_dim=self.bond_embedding_dim,
            atom_embedding_dim=self.atom_embedding_dim,
            state_embedding_dim=self.state_embedding_dim,
            rbf_type=self.rbf_type,
        )
        config.update(**self.kwargs)
        return config


@register
class GraphFieldEmbedding(GraphUpdateFunc):
    """
    Embedding the categorical field of a graph to continuous space
    """

    def __init__(self, nvocal: int = 95, embedding_dim: int = 16, field: str = "atoms", **kwargs):
        """

        Args:
            nvocal (int): number of vocabulary
            embedding_dim (int): the embedding dimension
            field (str): graph field for embedding
            **kwargs:
        """
        self.nvocal = nvocal
        self.embedding_dim = embedding_dim
        self.field = field
        self.kwargs = kwargs
        self.embedding = Embedding(nvocal, embedding_dim, **kwargs)
        super().__init__(update_func=self.embedding, update_field=field, **kwargs)

    def get_config(self) -> dict:
        """
        Get config dict for serialization
        Returns:
        """
        return {
            "nvocal": self.nvocal,
            "embedding_dim": self.embedding_dim,
            "field": self.field,
            "name": self.name,
            "trainable": self.trainable,
            "dtype": self.dtype,
        }
