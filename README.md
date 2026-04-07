# rock-vegetation-separation
Multi-scale geometric feature-based rock–vegetation separation and surface reconstruction for vegetation-occluded point clouds.
# Multi-Scale Geometric Feature-Based Rock–Vegetation Separation

This repository provides the implementation of the method proposed in our paper:

**“Multi-Scale Geometric Feature-Based Rock–Vegetation Separation in Vegetation-Occluded Point Clouds”**

The framework aims to achieve robust separation of rock and vegetation in complex terrestrial laser scanning (TLS) point clouds and supports subsequent rock surface reconstruction and geometric analysis.

---

##  Overview

Vegetation occlusion significantly degrades the geometric integrity of rock mass point clouds, making reliable surface reconstruction challenging.  
This work proposes a multi-scale geometric feature-driven framework that integrates:

- Multi-scale curvature analysis  
- Point density estimation  
- Elevation-based discrimination  
- Unsupervised clustering (K-means)  
- Spatial refinement via DBSCAN  
- Geometry-aware surface reconstruction  

The method demonstrates strong robustness and stability under vegetation-dominated conditions.

---

##  Features

- Multi-scale geometric feature extraction  
- Unsupervised rock–vegetation separation  
- Spatial connectivity optimization  
- Improved geometric consistency  
- Surface reconstruction with reduced artifacts  

---

##  Requirements

- Python 3.8 or higher  
- Open3D  
- NumPy  
- Scikit-learn  

Install dependencies:

```bash
pip install numpy open3d scikit-learn
