[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/m3gnet)](https://github.com/materialsvirtuallab/m3gnet/blob/main/LICENSE)[![Linting](https://github.com/materialsvirtuallab/m3gnet/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/m3gnet/workflows/Linting/badge.svg)[![Testing](https://github.com/materialsvirtuallab/m3gnet/workflows/Testing%20-%20main/badge.svg)](https://github.com/materialsvirtuallab/m3gnet/workflows/Testing/badge.svg)[![Downloads](https://pepy.tech/badge/m3gnet)](https://pepy.tech/project/m3gnet)

# M3GNet

M3GNet is a new materials graph neural network architecture that incorporates 3-body interactions. A key difference with prior
materials graph implementations such as [MEGNet](https://github.com/materialsvirtuallab/megnet) is the addition of the
coordinates for atoms and the 3×3 lattice matrix in crystals, which are necessary for obtaining tensorial quantities such as
forces and stresses via auto-differentiation. 

As a framework, M3GNet has diverse applications, including:
- Interatomic potential development. With the same training data, M3GNet performs similarly to state-of-the-art machine
  learning interatomic potentials (ML-IAPs). However, a key feature of a graph-based potential is its flexibility to
  scale to diverse chemical spaces. One of the key accomplishments of M3GNet is the development of a *universal IAP*
  that can work across the entire periodic table of the elements by training on relaxations performed in the Materials
  Project.
- Surrogate models for property predictions. Like the previous MEGNet architecture, M3GNet can be used to develop
  surrogate models for property predictions, achieving in many cases accuracies that better or similar to other state
  of the art ML models.

For detailed performance benchmarks, please refer to the publication in the [References](#references) section. The 
Sphinx-generated API documentation is available via the [Github Page](http://materialsvirtuallab.github.io/m3gnet).

# Table of Contents
* [System requirements](#system-requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Model training](#model-training)
* [Datasets](#datasets)
* [References](#references)

# System requirements

Inferences using the pre-trained models can be ran on any standard computer. For model training, the GPU memory needs 
to be > 18 Gb for a batch size of 32 using the crystal training data. In our work, we used a single RTX 3090 
GPU for model training. 

# Installation

M3GNet can be installed via pip:

```
pip install m3gnet
```

You can also directly download the source from Github and install from source.

## Apple Silicon Installation

Special care may be needed for Apple Silicon (M1, M1 Pro, M1 Max, M1 Ultra) machines. Apple Silicon has extremely
powerful ML capabilities. Here are the recommended steps to get m3gnet working on Apple Silicon devices.

1. Ensure that you already have XCode and CLI installed.
2. Install Miniconda or Anaconda.
3. Create a Python 3.9 environment,
```bash
conda create --name m3gnet python=3.9
conda activate m3gnet
```
4. First install tensorflow for Apple Silicon.
```bash
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal  # This is optional. If you encounter weird tensorflow errors, try uninstalling this first.
```
5. Install m3gnet but ignore dependencies (otherwise, pip will look for tensorflow).
```bash
pip install --no-deps m3gnet
```
6. You may also need to install `protobuf==3.20.0` and other dependencies like pymatgen, etc. manually.

# Usage

## Structure relaxation

A M3Gnet universal potential for the periodic table has been developed using data from Materials Project relaxations
since 2012. This universal potential can be used to perform structural relaxation of any arbitrary crystal as follows.

```python
from pymatgen.core import Structure, Lattice
from m3gnet.models import Relaxer

# Init a Mo structure with stretched lattice (DFT lattice constant ~ 3.168)
mo = Structure(Lattice.cubic(3.3), 
               ["Mo", "Mo"], [[0., 0., 0.], [0.5, 0.5, 0.5]])

relaxer = Relaxer()  # This loads the default pre-trained model

relax_results = relaxer.relax(mo)

final_structure = relax_results['final_structure']
final_energy = relax_results['trajectory'].energies[-1] / 2

print(f"Relaxed lattice parameter is {final_structure.lattice.abc[0]: .3f} Å")
print(f"Final energy is {final_energy.item(): .3f} eV/atom")
```

We will see the following output: 
```
Relaxed lattice parameter is  3.169 Å
Final energy is -10.859 eV/atom
```
The original lattice parameter of 
`3.3 Å` was successfully relaxed to `3.169 Å`, close to the DFT value of `3.168 Å`. 

The final energy -10.859 eV/atom is also close to DFT value of [-10.8456 
eV/atom](https://materialsproject.org/materials/mp-129/).

The relaxation takes less than 20 seconds on a single laptop.

## Molecular dynamics

Similarly the universal IAP can be used to perform molecular dynamics (MD) simulations as well.

```python
from pymatgen.core import Structure, Lattice
from m3gnet.models import MolecularDynamics

# Init a Mo structure with stretched lattice (DFT lattice constant ~ 3.168)
mo = Structure(Lattice.cubic(3.3), 
               ["Mo", "Mo"], [[0., 0., 0.], [0.5, 0.5, 0.5]])

md = MolecularDynamics(
    atoms=mo,
    temperature=1000,  # 1000 K
    ensemble='nvt',  # NVT ensemble
    timestep=1, # 1fs,
    trajectory="mo.traj",  # save trajectory to mo.traj
    logfile="mo.log",  # log file for MD
    loginterval=100,  # interval for record the log
)

md.run(steps=1000)
```

After the run, `mo.log` contains thermodynamic information similar to the following:

```bash
Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
0.0000         -21.3307     -21.3307       0.0000     0.0
0.1000         -21.3307     -21.3307       0.0000     0.0
0.2000         -21.2441     -21.3087       0.0645   249.7
0.3000         -21.0466     -21.2358       0.1891   731.6
0.4000         -20.9702     -21.1149       0.1447   559.6
0.5000         -20.9380     -21.1093       0.1713   662.6
0.6000         -20.9176     -21.1376       0.2200   850.9
0.7000         -20.9016     -21.1789       0.2773  1072.8
0.8000         -20.8804     -21.1638       0.2835  1096.4
0.9000         -20.8770     -21.0695       0.1925   744.5
1.0000         -20.8908     -21.0772       0.1864   721.2
```

The MD run takes less than 1 minute. 

# Model training

You can also train your own IAP using the `PotentialTrainer` in `m3gnet.trainers`. The training dataset can include:
- structures, a list of pymatgen Structures
- energies, a list of energy floats with unit `eV`.
- forces, a list of nx3 force matrix with unit `eV/Å`, where `n` is the number of atom in 
  each structure. `n` does not need to be the same for all structures. 
- stresses, a list of 3x3 stress matrices with unit `GPa` (optional)

For the `stresses`, we use the convention that compressive stress gives negative values. Stresses obtained from VASP (unit kBar)
calculations should multiply by `-0.1` to work directly with the model.

We use validation dataset to select the stopping epoch number. The dataset has similar format as the training dataset. 

A minimal example of model training is shown below.

```python
from m3gnet.models import M3GNet, Potential
from m3gnet.trainers import PotentialTrainer

import tensorflow as tf

m3gnet = M3GNet(is_intensive=False)
potential = Potential(model=m3gnet)

trainer = PotentialTrainer(
    potential=potential, optimizer=tf.keras.optimizers.Adam(1e-3)
)

trainer.train(
    structures,
    energies,
    forces,
    stresses,
    validation_graphs_or_structures=val_structures,
    val_energies=val_energies,
    val_forces=val_forces,
    val_stresses=val_stresses,
    epochs=100,
    fit_per_element_offset=True,
    save_checkpoint=False,
)
```

# Datasets

The training data used to develop the universal M3GNet IAP is `MPF.2021.2.8` and is hosted on
[figshare](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599) with DOI `10.6084/m9.figshare.19470599`.


# Reference

Please cite the following work:

```
Chi Chen, and Shyue Ping Ong. "A Universal Graph Deep Learning Interatomic Potential for the Periodic Table." 
arXiv preprint [arXiv:2202.02450](https://arxiv.org/abs/2202.02450) (2022).
```

# Acknowledgements

This work was primarily supported by the Materials Project, funded by the U.S. Department of Energy, Office of Science, 
Office of Basic Energy Sciences, Materials Sciences and Engineering Division under contract no. 
DE-AC02-05-CH11231: Materials Project program KC23MP. This work used the Expanse supercomputing cluster at the Extreme
Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number
ACI-1548562.
