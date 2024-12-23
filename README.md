# CPPU: Multi-Agent Learning for CSI Acquisition, Path Planning, and Updating

## Overview
This repository contains the implementation of the **CPPU algorithm** described in our paper: *"Policy Space Diversity for Informative Path Planning and GAI-enabled Updating CSI in ISAC"*. The CPPU algorithm leverages **multi-agent learning** and **Generative Artificial Intelligence (GAI)** to optimize **Channel State Information (CSI)** acquisition and localization for **Integrated Sensing and Communication (ISAC)** in 6G networks.

## Abstract
Integrated Sensing and Communication (ISAC) is a core component of 6G technology, driving advancements in networking and communication in combination with Generative Artificial Intelligence (GAI). By utilizing GAI’s predictive functions, indoor positioning achieves higher efficiency through optimized fingerprint localization methods.

Traditional approaches to CSI collection are costly and lack adaptability in dynamic environments. To address these challenges, CPPU combines **multi-agent learning** with a **GAI model** to achieve efficient CSI acquisition, path planning, and data updating.

Key contributions include:
- Partitioning the terrain into multiple regions, ensuring **comprehensive path coverage** with no backtracking.
- Using a dynamic programming strategy (policy space response oracle) to identify **informative paths** within each region, integrating full-coverage paths and small datasets of real CSI.
- Employing the GAI model to **predict and update CSI distribution** at unvisited locations, significantly reducing data acquisition costs while maintaining localization accuracy.

Experimental results in two real-world scenarios demonstrate CPPU’s effectiveness in reducing data acquisition costs and achieving competitive localization accuracy.

## Features
- **Multi-Agent Learning**: Collaborative path planning with multiple agents, each assigned to specific regions.
- **GAI-Powered CSI Updates**: Efficiently predict and update CSI values at unvisited locations.
- **Dynamic Programming for Path Optimization**: Identify the most informative paths using a policy space oracle.
- **Scalable and Adaptive**: Designed for dynamic environments with minimal manual intervention.

## Usage
The main files of this project are Lab_UpdateCSI.py and Meet_UpdateCSI.py. Run these files to execute the program.

## Results
Our experimental validation in two real-world scenarios shows:

- **Reduced data acquisition costs**:  
  CPPU achieves significant efficiency improvements compared to traditional CSI collection methods.

- **Maintained localization accuracy**:  
  Despite cost reductions, CPPU achieves accuracy comparable to that of existing methods.

## Citation
If you find our work helpful, please cite:
```bibtex
@article{zhu2024cppu,
  title={CPPU: Policy Space Diversity for Informative Path Planning and GAI-enabled Updating CSI in ISAC},
  author={Zhu, X. and Liu, J. and Zhang, T. and others},
  journal={IEEE Transactions on Cognitive Communications and Networking},
  pages={1-14},
  year={2024},
  DOI={10.1109/TCCN.2024.3522088}
}
