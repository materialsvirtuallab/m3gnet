[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/m3gnet)](https://github.com/materialsvirtuallab/m3gnet/blob/main/LICENSE)[![Linting](https://github.com/materialsvirtuallab/m3gnet/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/m3gnet/workflows/Linting/badge.svg)[![Testing](https://github.com/materialsvirtuallab/m3gnet/workflows/Testing%20-%20main/badge.svg)](https://github.com/materialsvirtuallab/m3gnet/workflows/Testing/badge.svg)[![Downloads](https://pepy.tech/badge/m3gnet)](https://pepy.tech/project/m3gnet)

# M3GNet

M3GNet is a new materials graph neural network architecture that incorporates 3-body interactions. A key difference 
with prior materials graph implementations such as [MEGNet](https://github.com/materialsvirtuallab/megnet) is the
addition of the coordinates for atoms and the 3×3 lattice matrix in crystals, which are necessary for obtaining
tensorial quantities such as forces and stresses via auto-differentiation. 

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
4. First install tensorflow and its dependencies for Apple Silicon.
```bash
conda install -c apple tensorflow-deps
pip install tensorflow-macos
```
5. If you wish, you can install tensorflow-metal, which helps speed up training. If you encounter weird tensorflow 
   errors, you should uninstall tensorflow-metal and see if it fixes the errors first.
```
pip install tensorflow-metal 
```
6. Install m3gnet but ignore dependencies (otherwise, pip will look for tensorflow).
```bash
pip install --no-deps m3gnet
```
7. Install other dependencies like pymatgen, etc. manually.
```bash
pip install protobuf==3.20.0 pymatgen ase cython
```
8. Once you are done, you can try running `pytest m3gnet` to see if all tests pass.

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
The initial lattice parameter of 3.3 Å was successfully relaxed to 3.169 Å, close to the DFT value of 3.168 Å. 

The final energy -10.859 eV/atom is also close to Materials Project DFT value of
[-10.8456 eV/atom](http://materialsproject.org/materials/mp-129/).

The relaxation takes less than 20 seconds on a single laptop.

Here are more comprehensive benchmarks for cubic crystals based on 
[exp data on Wikipedia](http://en.wikipedia.org/wiki/Lattice_constant) and MP DFT data. The Jupyter notebook is in the 
[examples](/examples/Cubic%20Crystal%20Test.ipynb) folder.

| Material    |       a | Crystal structure   |   predicted_a |    mp_a | % error vs Expt   | % error vs MP   |
|:------------|--------:|:--------------------|--------------:|--------:|:------------------|:----------------|
| C (diamond) | 3.567   | Diamond (FCC)       |       3.5718  | 3.57371 | 0.13%             | -0.05%          |
| Si          | 5.43102 | Diamond (FCC)       |       5.45002 | 5.46873 | 0.35%             | -0.34%          |
| Ge          | 5.658   | Diamond (FCC)       |       5.7698  | 5.76286 | 1.98%             | 0.12%           |
| AlAs        | 5.6605  | Zinc blende (FCC)   |       5.73027 | 5.73376 | 1.23%             | -0.06%          |
| AlP         | 5.451   | Zinc blende (FCC)   |       5.50346 | 5.50711 | 0.96%             | -0.07%          |
| AlSb        | 6.1355  | Zinc blende (FCC)   |       6.22817 | 6.23376 | 1.51%             | -0.09%          |
| GaP         | 5.4505  | Zinc blende (FCC)   |       5.5054  | 5.5063  | 1.01%             | -0.02%          |
| GaAs        | 5.653   | Zinc blende (FCC)   |       5.75055 | 5.75018 | 1.73%             | 0.01%           |
| GaSb        | 6.0959  | Zinc blende (FCC)   |       6.21939 | 6.21906 | 2.03%             | 0.01%           |
| InP         | 5.869   | Zinc blende (FCC)   |       5.9679  | 5.95673 | 1.69%             | 0.19%           |
| InAs        | 6.0583  | Zinc blende (FCC)   |       6.25374 | 6.18148 | 3.23%             | 1.17%           |
| InSb        | 6.479   | Zinc blende (FCC)   |       6.63863 | 6.63322 | 2.46%             | 0.08%           |
| MgO         | 4.212   | Halite (FCC)        |       4.2567  | 4.25648 | 1.06%             | 0.01%           |
| CdS         | 5.832   | Zinc blende (FCC)   |       5.9419  | 5.94083 | 1.88%             | 0.02%           |
| CdSe        | 6.05    | Zinc blende (FCC)   |       6.20987 | 6.21283 | 2.64%             | -0.05%          |
| CdTe        | 6.482   | Zinc blende (FCC)   |       6.62619 | 6.62905 | 2.22%             | -0.04%          |
| ZnO         | 4.58    | Halite (FCC)        |       4.33424 | 4.33888 | -5.37%            | -0.11%          |
| ZnS         | 5.42    | Zinc blende (FCC)   |       5.45297 | 5.45027 | 0.61%             | 0.05%           |
| PbS         | 5.9362  | Halite (FCC)        |       6.01752 | 6.00645 | 1.37%             | 0.18%           |
| PbTe        | 6.462   | Halite (FCC)        |       6.56111 | 6.56567 | 1.53%             | -0.07%          |
| BN          | 3.615   | Zinc blende (FCC)   |       3.62485 | 3.626   | 0.27%             | -0.03%          |
| BP          | 4.538   | Zinc blende (FCC)   |       4.54711 | 4.54682 | 0.20%             | 0.01%           |
| LiF         | 4.03    | Halite              |       4.08531 | 4.08343 | 1.37%             | 0.05%           |
| LiCl        | 5.14    | Halite              |       5.14745 | 5.15275 | 0.15%             | -0.10%          |
| LiBr        | 5.5     | Halite              |       5.51076 | 5.51343 | 0.20%             | -0.05%          |
| LiI         | 6.01    | Halite              |       6.02709 | 6.0257  | 0.28%             | 0.02%           |
| NaF         | 4.63    | Halite              |       4.69553 | 4.69625 | 1.42%             | -0.02%          |
| NaCl        | 5.64    | Halite              |       5.69497 | 5.69169 | 0.97%             | 0.06%           |
| NaBr        | 5.97    | Halite              |       6.01922 | 6.0276  | 0.82%             | -0.14%          |
| NaI         | 6.47    | Halite              |       6.52739 | 6.532   | 0.89%             | -0.07%          |
| KF          | 5.34    | Halite              |       5.41971 | 5.42398 | 1.49%             | -0.08%          |
| KCl         | 6.29    | Halite              |       6.39634 | 6.38359 | 1.69%             | 0.20%           |
| KBr         | 6.6     | Halite              |       6.70797 | 6.70308 | 1.64%             | 0.07%           |
| KI          | 7.07    | Halite              |       7.18309 | 7.18534 | 1.60%             | -0.03%          |
| RbF         | 5.65    | Halite              |       5.76843 | 5.73892 | 2.10%             | 0.51%           |
| RbCl        | 6.59    | Halite              |       6.67994 | 6.69873 | 1.36%             | -0.28%          |
| RbBr        | 6.89    | Halite              |       6.98219 | 7.02793 | 1.34%             | -0.65%          |
| RbI         | 7.35    | Halite              |       7.61756 | 7.48785 | 3.64%             | 1.73%           |
| CsF         | 6.02    | Halite              |       6.1265  | 6.11801 | 1.77%             | 0.14%           |
| CsCl        | 4.123   | Caesium chloride    |       4.20308 | 4.20906 | 1.94%             | -0.14%          |
| CsI         | 4.567   | Caesium chloride    |       4.90767 | 4.66521 | 7.46%             | 5.20%           |
| Al          | 4.046   | FCC                 |       4.04108 | 4.03893 | -0.12%            | 0.05%           |
| Fe          | 2.856   | BCC                 |       2.85237 | 2.84005 | -0.13%            | 0.43%           |
| Ni          | 3.499   | FCC                 |       3.5086  | 3.5058  | 0.27%             | 0.08%           |
| Cu          | 3.597   | FCC                 |       3.61199 | 3.62126 | 0.42%             | -0.26%          |
| Mo          | 3.142   | BCC                 |       3.16937 | 3.16762 | 0.87%             | 0.06%           |
| Pd          | 3.859   | FCC                 |       3.95466 | 3.95707 | 2.48%             | -0.06%          |
| Ag          | 4.079   | FCC                 |       4.16702 | 4.16055 | 2.16%             | 0.16%           |
| W           | 3.155   | BCC                 |       3.18826 | 3.18741 | 1.05%             | 0.03%           |
| Pt          | 3.912   | FCC                 |       3.97714 | 3.97677 | 1.67%             | 0.01%           |
| Au          | 4.065   | FCC                 |       4.17431 | 4.17129 | 2.69%             | 0.07%           |
| Pb          | 4.92    | FCC                 |       5.02849 | 5.05053 | 2.21%             | -0.44%          |
| V           | 3.0399  | BCC                 |       2.99346 | 2.99254 | -1.53%            | 0.03%           |
| Nb          | 3.3008  | BCC                 |       3.32221 | 3.32052 | 0.65%             | 0.05%           |
| Ta          | 3.3058  | BCC                 |       3.31741 | 3.32229 | 0.35%             | -0.15%          |
| TiN         | 4.249   | Halite              |       4.25254 | 4.25353 | 0.08%             | -0.02%          |
| ZrN         | 4.577   | Halite              |       4.61602 | 4.61762 | 0.85%             | -0.03%          |
| HfN         | 4.392   | Halite              |       4.53838 | 4.53774 | 3.33%             | 0.01%           |
| VN          | 4.136   | Halite              |       4.1281  | 4.12493 | -0.19%            | 0.08%           |
| CrN         | 4.149   | Halite              |       4.16068 | 0       | 0.28%             | inf%            |
| NbN         | 4.392   | Halite              |       4.45474 | 4.45247 | 1.43%             | 0.05%           |
| TiC         | 4.328   | Halite              |       4.33493 | 4.33565 | 0.16%             | -0.02%          |
| ZrC0.97     | 4.698   | Halite              |       4.72451 | 4.72434 | 0.56%             | 0.00%           |
| HfC0.99     | 4.64    | Halite              |       4.65023 | 4.65131 | 0.22%             | -0.02%          |
| VC0.97      | 4.166   | Halite              |       4.16476 | 4.16195 | -0.03%            | 0.07%           |
| NC0.99      | 4.47    | Halite              |       3.67761 | 0       | -17.73%           | inf%            |
| TaC0.99     | 4.456   | Halite              |       4.48225 | 4.48208 | 0.59%             | 0.00%           |
| ScN         | 4.52    | Halite              |       4.51797 | 4.51831 | -0.04%            | -0.01%          |
| KTaO3       | 3.9885  | Cubic perovskite    |       4.03265 | 4.03084 | 1.11%             | 0.05%           |
| SrTiO3      | 3.98805 | Cubic perovskite    |       3.94481 | 3.94513 | -1.08%            | -0.01%          |
| EuTiO3      | 7.81    | Cubic perovskite    |       3.92943 | 3.96119 | -49.69%           | -0.80%          |
| SrVO3       | 3.838   | Cubic perovskite    |       3.90604 | 3.90089 | 1.77%             | 0.13%           |
| CaVO3       | 3.767   | Cubic perovskite    |       3.83451 | 3.83041 | 1.79%             | 0.11%           |

As you can see, almost all cubic lattice constants are within 1% of the DFT values. The only major errors are with
EuTiO3 wrt to the Wikipedia value. It is quite likely the Wikipedia value is wrong. All relaxations take less than 1s
on a M1 Max Mac.

## Molecular dynamics

Similarly, the universal IAP can be used to perform molecular dynamics (MD) simulations as well.

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
- energies, a list of energy floats with unit eV.
- forces, a list of nx3 force matrix with unit eV/Å, where n is the number of atom in 
  each structure. n does not need to be the same for all structures. 
- stresses, a list of 3x3 stress matrices with unit GPa (optional)

For stresses, we use the convention that compressive stress gives negative values. Stresses obtained from
VASP calculations (default unit is kBar) should be multiplied by -0.1 to work directly with the model.

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
