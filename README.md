# m3gnet
A universal material graph interatomic potential with three-body interactions

# Table of Contents
* [Introduction](#introduction)
* [System requirements](#systemreq)
* [Installation](#installation)
* [Demo](#demo)
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
This package is supported for macOS and Linux. The package has been tested on the following systems:

- macOS: Monterey 12.1 
- Linux: Ubuntu 18.04 (with tensorflow==2.7.0)


The software was built and tested on MacOS and Linux OS systems. 

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

## Molecular Dynamics

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

```angular2html
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

<a name="datasets"></a>
#Datasets
The training data `MPF.2021.2.8` is hosted on figshare.


<a name="references"></a>

# Reference
This package is the result from our recent [paper](https://arxiv.org/abs/2202.02450)
```angular2html
Chen, Chi, and Shyue Ping Ong. "A Universal Graph Deep Learning Interatomic Potential for the Periodic Table." arXiv preprint arXiv:2202.02450 (2022).
```