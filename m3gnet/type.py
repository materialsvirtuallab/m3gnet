"""Define several types for convenient use"""

from typing import Any, Callable, List, Optional, Union

import numpy as np
from ase.atoms import Atoms
from pymatgen.core.structure import Molecule, Structure

OptStrOrCallable = Optional[Union[str, Callable[..., Any]]]
StructureOrMolecule = Union[Structure, Molecule, Atoms]
VectorLike = Union[List[float], np.ndarray]
