# Chemometrix Documentation

This document outlines the methodology used in the SVR Bioactivity Dashboard for visualizing and analyzing chemical data.

## 1. Data Acquisition and Preprocessing

- Bioactivity data for a specific ChEMBL Target ID is fetched from the ChEMBL database. The data is filtered to include common bioactivity standard types such as IC50, Ki, and EC50.
- The raw data is preprocessed to remove entries with missing standard values or canonical SMILES strings.
- A **pIC50** value is calculated for each compound, which represents its potency. This is calculated as `-log10(IC50)`, where the IC50 value is converted to molar concentration.

## 2. Fingerprint Generation and Similarity

To analyze the structural similarity of the compounds, we use a combination of 2D and 3D fingerprinting techniques.

### 2.1 2D Morgan Fingerprints

- For each compound, a **2D Morgan fingerprint** is generated using RDKit. This is a type of circular fingerprint that encodes the local structural environment of each atom.
- These 2D fingerprints are used for two main purposes:
    1. **Clustering:** The fingerprints are clustered using the **KMeans** algorithm. This groups the molecules into structurally similar clusters.
    2. **Dimensionality Reduction:** The high-dimensional fingerprint space is reduced to two dimensions using **Principal Component Analysis (PCA)**.

### 2.2 3D Flexibility Score

- In addition to the 2D fingerprints, a **3D flexibility score** is calculated for each molecule.
- This is done by generating a single low-energy conformer for each molecule and then calculating the **Normalized Principal Moments of Inertia (NPR)**.
- This score provides a measure of the molecule's 3D shape, indicating whether it is more "rod-like" or "sphere-like".

## 3. SVR Modeling and Visualization

A Support Vector Regression (SVR) model is trained to predict the pIC50 of a compound based on its 2D Morgan fingerprint. The results are visualized in two interactive plots:

### 3.1 Bioactivity Landscape Visualization

- This plot displays the chemical space of the compounds, with the x and y axes representing the first two principal components (PC1 and PC2) from the PCA of the Morgan fingerprints.
- The color of each point corresponds to its **KMeans cluster**, allowing for the visualization of structurally similar groups.
- The size of each point is proportional to the **SVR prediction error**, highlighting compounds that the model struggled to predict accurately.
- Hovering over a point reveals detailed information, including its ChEMBL ID, SMILES string, pIC50 values (actual and predicted), cluster, and 3D flexibility score.

### 3.2 SVR Performance Analysis

- This plot shows the direct relationship between the **Actual pIC50** and the **Predicted pIC50**.
- It includes the **SVR "tube"** (defined by the `epsilon` parameter), which is the region where prediction errors are not penalized by the model.
- The background is colored to represent regions of **low, medium, and high potency**, providing a visual guide to the potency landscape.
- The points are colored based on whether they fall **inside or outside** the SVR tube, giving a clear indication of the model's performance for each prediction.
