# Adaptive Grid Discretizations using Lattice Basis Reduction (AGD-LBR)
## A set of tools for discretizing anisotropic PDEs on cartesian grids

This repository contains
- the agd library (Adaptive Grid Discretizations)
- a series of *jupyter notebooks* in the Python&reg; language, reproducing my research in Anisotropic PDE discretizations and their applications.

### The AGD library

The recommended way to install is
```console
conda install agd -c agd-lbr --force
```

### The notebooks

The notebooks are intended as documentation and testing for the adg library. They encompass:
* Anisotropic fast marching methods, for shortest path computation.
* Non-divergence form PDEs, including non-linear PDEs such as Monge-Ampere.
* Divergence form anisotropic PDEs, often encountered in image processing.
* Algorithmic tools, related with lattice basis reduction methods, and automatic differentiation.

The notebooks can be visualized online, [view summary online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Summary.ipynb
), or executed and/or modified offline.
For offline consultation, please download and install [anaconda](https://www.anaconda.com) or [miniconda](https://conda.io/en/latest/miniconda.html).  
*Optionally*, you may create a dedicated conda environnement by typing the following in a terminal:
```console
conda env create --file agd-hfm.yaml
conda activate agd-hfm
```
In order to open the book summary, type in a terminal:
```console
jupyter notebook Summary.ipynb
```
Then use the hyperlinks to navigate within the notebooks.

<!---
All terminal commands presented here assume that the base directory is the directory containing this file.

### Anisotropic Fast Marching methods

In folder *Notebooks_FMM*. A series of notebooks illustrating the Hamilton-Fast-Marching (HFM) library, which is devoted to solving shortest path problems w.r.t. anisotropic metrics. These notebooks are intended as documentation, user's guide, and test cases for the HFM library.

You can view the summary of this series [online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks_FMM/Summary.ipynb), or open it offline with the following terminal command:
```console
jupyter notebook Notebooks_FMM/Summary.ipynb
```

In order to run these notebooks, you need the binaries of the HFM library. It is open source and available on the following [Github repository](https://github.com/mirebeau/AdaptiveGridDiscretizations)

### Non-linear second order PDEs in non-divergence form

In folder *Notebooks_NonDiv*. This collection of notebooks presents a series of general principles and reference implementations for *Non-linear  Partial Differential Equations (PDEs) in non-divergence form*, using *adaptive finite difference schemes on cartesian grids*.

You can view the summary of this series [online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks_NonDiv/Summary.ipynb), or open it offline with the following terminal command:
```console
jupyter notebook Notebooks_NonDiv/Summary.ipynb
```

### Anisotropic PDEs in divergence form

In folder *Notebooks_Div*. This collection of notebooks illustrates the discretization of *anisotropic PDEs in divergence form*, using non-negative discretizations which obey the discrete maximum principle.

You can view the summary of this series [online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks_Div/Summary.ipynb), or open it offline with the following terminal command:
```console
jupyter notebook Notebooks_Div/Summary.ipynb
```
--->
