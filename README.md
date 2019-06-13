# Adaptive grid discretizations
## A set of tools for discretizing anisotropic PDEs on cartesian grids

This repository contains two series of *jupyter notebooks* in the Python&reg; language, reproducing my research in Anisotropic PDE discretizations and their applications. They can be visualized online, or executed and/or modified offline.

For offline consultation, please download and install [anaconda](https://www.anaconda.com) or [miniconda](https://conda.io/en/latest/miniconda.html).  
Optionally, you may create a conda environnement dedicated to their visualization by typing the following in a terminal
```console
conda env create --file Notebooks_FMM/hfm-jupyter-mayavi.yaml
source activate hfm-jupyter-mayavi
```
All terminal commands presented here assume that the base directory is the directory containing this file.

### Anisotropic Fast Marching methods

In folder *Notebooks_FMM*. A series of notebooks illustrating the Hamilton-Fast-Marching (HFM) library, which is devoted to solving shortest path problems w.r.t. anisotropic metrics. These notebooks are intended as documentation, user's guide, and test cases for the HFM library.

You can view the summary of this series [online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks_FMM/Summary.ipynb), or open it offline with the following terminal command:
```console
jupyter notebook Notebooks_FMM/Summary.ipynb
```

In order to run these notebooks, you need the binaries of the HFM library. It is open source and available on the following [Github repository](https://github.com/mirebeau/AdaptiveGridDiscretizations)

### Monotone discretizations of Anisotropic PDEs

In folder *Notebooks_PDE*. This collection of notebooks presents a series of general principles and reference implementations for *anisotropic Partial Differential Equations* (PDEs), using *adaptive finite difference schemes on cartesian grids*.

You can view the summary of this series [online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks_PDE/Summary.ipynb), or open it offline with the following terminal command:
```console
jupyter notebook Notebooks_PDE/Summary.ipynb
```
