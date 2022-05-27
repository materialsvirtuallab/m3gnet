[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/m3gnet)](https://github.com/materialsvirtuallab/m3gnet/blob/main/LICENSE)[![Linting](https://github.com/materialsvirtuallab/m3gnet/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/m3gnet/workflows/Linting/badge.svg)[![Testing](https://github.com/materialsvirtuallab/m3gnet/workflows/Testing%20-%20main/badge.svg)](https://github.com/materialsvirtuallab/m3gnet/workflows/Testing/badge.svg)[![Downloads](https://pepy.tech/badge/m3gnet)](https://pepy.tech/project/m3gnet)

# m3gnet
A universal material graph interatomic potential with three-body interactions

# Table of Contents
* [Introduction](#introduction)
* [System requirements](#systemreq)
* [Installation](#installation)
* [Demo](#demo)
* [Model training](#training)
* [Datasets](#datasets)
* [References](#references)

<a name="introduction"></a>
# Introduction

This repository contains the `M3GNet` interatomic potential for the periodic 
table. The model has been developed for inorganic crystals using the 
Materials Project relaxation trajectory as training data.


<a name="systemreq"></a>
# System requirements
## Hardware requirements
Inferences using the pre-trained models can be ran on any standard computer.
For model training, the GPU memory needs to be > 18 Gb for a batch size of 
32 using the crystal training data. In our work, we used single RTX 3090 
GPU for model training. 

## Software requirements
The package has been tested on the following systems:

- macOS: Monterey 12.1 
- Linux: Ubuntu 18.04 (with tensorflow==2.7.0)
- Windows: 11


<a name="Installation"></a>
# Installation

The following dependencies are needed 

```
pymatgen==2022.2.10
pandas==1.4.1
tensorflow==2.7.0
numpy==1.22.2
monty==2022.3.12
sympy==1.9
ase==3.22.0
cython==0.29.26
```

`m3gnet` can be installed via source code installation by cloning the code from github

```
git clone https://github.com/materialsvirtuallab/m3gnet.git
cd m3gnet
pip install -r requirements.txt
python setup.py install
```

The installation time should be less than 1 minute.

## Apple Silicon Installation

Apple silicon has extremely powerful ML capabilities. But there are some special installation requirements.
Here are the recommended steps to get m3gnet working on Apple Silicon devices.

1. Ensure that you already have XCode and CLI installed.
2. Install Miniconda. 
3. Create an Python 3.9 environment,
```bash
conda create --name m3gnet python=3.9
conda activate m3gnet
```
4. First install tensorflow with the metal plugin for Apple Silicon.
```bash
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```
5. Install m3gnet.
```bash
pip install m3gnet
```

<a name="demo"></a>
# Demo

## Structure relaxation

Here is an example of how to run `m3gnet` relaxation of any crystals.

```python
from pymatgen.core import Structure, Lattice
from m3gnet.models import Relaxer

# Init a Mo structure with stretched lattice (DFT lattice constant ~ 3.168)
mo = Structure(Lattice.cubic(3.3), 
               ["Mo", "Mo"], [[0., 0., 0.], [0.5, 0.5, 0.5]])

relaxer = Relaxer()  # this loads the default model

relax_results = relaxer.relax(mo)

final_structure = relax_results['final_structure']
final_energy = relax_results['trajectory'].energies[-1] / 2

print(f"Relaxed lattice parameter is {final_structure.lattice.abc[0]: .3f} Å")
print(f"Final energy is {final_energy.item(): .3f} eV/atom")
```
We will see output 
```
Relaxed lattice parameter is  3.169 Å
Final energy is -10.859 eV/atom
```
The original lattice parameter of 
`3.3 Å` was successfully relaxed to `3.169 Å`, close to the DFT value of `3.168 Å`. 

The final energy `-10.859 eV/atom` is also close to DFT value of [`-10.8456 
eV/atom`](https://materialsproject.org/materials/mp-129/).

The relaxation takes less than 20 seconds.

## Molecular dynamics

Molecular dynamics simulations can be easily performed as well.

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

After the run, `mo.log` contains thermodynamic information similar to the 
following 

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

<a name="training"></a>
# Model training

The potential model can be trained using the `PotentialTrainer` in `m3gnet.trainers`.
The training dataset can be

- structures, a list of pymatgen Structures
- energies, a list of energy floats
- forces, a list of nx3 force matrix, where `n` is the number of atom in 
  each structure. `n` does not need to be the same for all structures.
- stresses, a list of 3x3 stress matrix. 

where the `stresses` is optional. 

For the `stresses`, we use the convention that compressive stress gives 
negative values. `VASP` stress should change signs to work directly with 
the model.

We use validation dataset to select the stopping epoch number. The dataset 
has similar format as the training dataset. 

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
<a name="datasets"></a>
# Datasets
The training data `MPF.2021.2.8` is hosted on [figshare](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599) 
with DOI `10.6084/m9.figshare.19470599`.


<a name="references"></a>

# Reference
This package is the result from our recent [paper](https://arxiv.org/abs/2202.02450)
```angular2html
Chi Chen, and Shyue Ping Ong. "A Universal Graph Deep Learning Interatomic Potential for the Periodic Table." arXiv preprint arXiv:2202.02450 (2022).
```
