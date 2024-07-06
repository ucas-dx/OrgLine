<p align="center">
  <img src="pipeline.png" alt="OrgLine Logo" width="200">
</p>

## Introduction
OrgLine, the first multitask analysis pipeline for organoids, which leverages a pretrained detector and a vision foundational model to facilitate large-scale automated cultivation and analysis of organoids. OrgLine achieves precise localization, counting, classification, and segmentation of organoids, providing a foundational and efficient analytical support system.

## Getting Started
 <img src="https://camo.githubusercontent.com/eead68545f3aecadceca3a53cc6adcf6191e11ba8d598feab7f00615fa4e95ea/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f756c7472616c79746963733f6c6f676f3d707974686f6e266c6f676f436f6c6f723d676f6c64" alt="python version">
 
- Environment Setup Configuration
  
  Automatic environment setup, please be patient.
  ```bash
  conda create -n orgline python=3.10
  conda activate orgline
  conda install git
  git clone https://github.com/ucas-dx/OrgLine.git
  cd OrgLine
  python install_env.py
  conda install conda-forge::vs2015_runtime 
  ```
- Quick usage
  ```bash
  cd models & python simple_inference.py
  ```
- Jupyter

  You can run the contents of the [SimpleOperation.ipynb](https://github.com/ucas-dx/OrgLine/blob/main/SimpleOperation.ipynb) file, and all processes will be automated.
  ```bash
  from models.simple_inference import OrgAnalysis
  image_folder = 'images'
  org_analysis = OrgAnalysis(image_folder)
  org_analysis.analyze(show_seg=True,show_bboxes=False)
  ```
  <p align="center">
  <img src="https://github.com/ucas-dx/OrgLine/blob/main/output.png" alt="Output Image">
  </p>

  
 



