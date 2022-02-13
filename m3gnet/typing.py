# -*- coding: utf-8 -*-
"""Define several typing for convenient use"""

from typing import Union, Callable, Optional, Any, List

import numpy as np
from pymatgen.core import Structure, Molecule
from ase.atoms import Atoms

OptStrOrCallable = Optional[Union[str, Callable[..., Any]]]
StructureOrMolecule = Union[Structure, Molecule, Atoms]
VectorLike = Union[List[float], np.ndarray]
