# Hybrid Quantum-Classical Neural Network for Spam Email Detection

## Overview
We are excited to present a project that leverages quantum computing and machine learning to detect spam email effectively.

![Quantum_Spam_diagram.png](Quantum_Spam_diagram.png "pipeline")


## Installation & Usage

1. To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/zhangl64/qnn-spam-detect.git
```
```
cd qnn-spam-detect
```
2. Create a new virtual environment with Python **3.9.20** version: 
```
conda create -n qml python=3.9.20 anaconda
```
3. Activate the new environment: 
```
conda activate qml
```
4. Install the additional Python packages required:

```
python3 -m pip install -r requirements.txt
```
OR if anaconda installed, use 
```
conda env create -f environment.yml
```

5. Run the first two cells of ```dataPrep.ipynb ```to install the data.

Please update the dataset directory path in main.py to match your local setup.

6. Run the quantum model using the following command:

```
python main.py
```
<!--
## Citation
If you use this project or its findings, please cite it as follows:

```

```
-->


## Contact
For any inquiries, please reach out via email at ainazj1@umbc.edu and dkm26@umbc.edu
