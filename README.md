# CellMap - RNA landscape analysis framework

<div style="text-align:left"><img style="width:100%; height: auto" src="https://github.com/yusuke-imoto-lab/CellMap/blob/main/images/CellMap_overview.jpg"/></div>

The CellMap is a RNA landscape analysis framework for scRNA-seq data. 

---

## Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Requirements](#requirements)
* [Tutorial](#tutorial)
* [Visualization (CellMapViewer)](#visualization)
* [License](#license)
<!-- * [Citation](#citation) -->
* [Contact](#contact)

---

## Overview

**CellMap** is a framework for reconstructing and analyzing the transcriptional landscape of single cells using scRNA-seq data and inferred velocity fields. The main components are:

1. **Input**  
   - **scRNA-seq count matrix** \(X ∈ ℤ_{≥0}^{n×d}\)  
     - \(n\): number of cells  
     - \(d\): number of genes  
   - **Velocity matrix** \(V ∈ ℝ^{n×d}\)  
     - Any velocity-estimation method (e.g. RNA velocity)

2. **Hodge Decomposition**  
   Decompose the cell-state velocity field \(V\) into three orthogonal components:  
   - **Gradient (potential) flow**  
   - **Divergence-free (cyclic) flow**  
   - **Harmonic flow**

3. **Scalar Potentials**  
   - **Potential** \(\phi ∈ ℝ^n\)  
     - Represents cellular potency (the “height” of the landscape)  
   - **Orthogonal potential** \(\psi ∈ ℝ^n\)  
     - Orthogonal component satisfying \(\nabla\phi \cdot \nabla\psi = 0\)

4. **Outputs**  
   - **RNA Landscape**  
     - 3D surface embedding of the potential field, mimicking Waddington’s landscape  
   - **Gene Regulatory Dynamics**  
     - Flow vectors highlighting where a gene regulates on the landscape  
   - **Single-cell Trajectories**  
     - Pseudotemporal paths of individual cells on the landscape  
   - **DEG Dynamics**  
     - Time variations of differentially expressed genes

---

## Installation

You can install **cellmap** either from PyPI or directly from the GitHub repository:

**From PyPI:**

```bash
pip install cellmap
```

**From GitHub:**

```bash
git clone https://github.com/yusuke-imoto-lab/CellMap.git
cd eps-attracting-basin
pip install -r requirements.txt
```

---

## Requirements
* Python3
* matplotlib
* numpy
* scipy
* scikit-learn
* adjustText
* anndata
* plotly
* scanpy
* scvelo
* nbformat
* lingam

---

## Turtorial

Turtorials are prepared by jupyter notebook.

* [Tutorial for pancreas endocrine cell data](https://github.com/yusuke-imoto-lab/CellMap/blob/main/tutorial/CellMap_tutorial_pancreas.ipynb)

---

## Visualization

We prepare original visualization system *[CellMap viewer](https://github.com/yusuke-imoto-lab/CellMapViewer)*. 

<div style="text-align:left"><img style="width:100%; height: auto" src="https://github.com/yusuke-imoto-lab/CellMapViewer/blob/main/Images/TopImage.png"/></div>


---

## License

MIT © 2025 Yusuke Imoto

---

## Contact

* **Yusuke Imoto**
* Email: [imoto.yusuke.4e@kyoto-u.ac.jp](mailto:imoto.yusuke.4e@kyoto-u.ac.jp)
* GitHub: [yusuke-imoto-lab/CellMap](https://github.com/yusuke-imoto-lab/CellMap)
