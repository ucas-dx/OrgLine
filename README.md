<p align="center">
  <img src="pipeline.png" alt="OrgLine Logo" width="200">
</p>

## Introduction
OrgLine, the first multitask analysis pipeline for organoids, which leverages a pretrained detector and a vision foundational model to facilitate large-scale automated cultivation and analysis of organoids. OrgLine achieves precise localization, counting, classification, and segmentation of organoids, providing a foundational and efficient analytical support system.

## Getting Started
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)
- Environment Setup Configuration
```bash
conda create -n orgline python=3.10
conda activate orgline
conda install git
git clone https://github.com/ucas-dx/OrgLine.git
cd OrgLine
python install_env.py
```
- Quick usage
```bash
python models/simple_inference.py
```
- Jupyter
You can run the contents of the [SimpleOperation.ipynb](https://github.com/ucas-dx/OrgLine/blob/main/SimpleOperation.ipynb) file, and all processes will be automated.
```bash
from models.simple_inference import OrgAnalysis
image_folder = 'images'
org_analysis = OrgAnalysis(image_folder)
org_analysis.analyze()
```
  
 



