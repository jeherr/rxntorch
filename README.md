# RxnTorch

A graph neural network model for predicting reaction products. Based on "A graph-convolutional neural network model for the prediction of chemical reactivity." (DOI: 10.1039/C8SC04228D) 

## Installation

Using Anaconda is the easiest way, as RDKit is a required dependency which must be installed with conda, or built from source. First [install Anaconda](https://docs.anaconda.com/anaconda/install/). 

Next, it is recommended to create a conda environment specifically for this project. The code is developed using Python 3.6, and hasn't been tested for any other versions at this time. Create a conda environment with
```
conda create --name rxntorch python=3.6
```
Then, activate the new conda environment with
```
conda activate rxntorch
```
Next, install RDKit
```
conda install -c rdkit rdkit 
```
Installing PyTorch depends on whether you will be using CUDA or not. For a CUDA enabled version
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
otherwise, for a CPU only version
```
conda install pytorch torchvision cpuonly -c pytorch
```
Next, install tqdm and scikit-learn with
```
conda install tqdm scikit-learn
```
Finally, clone this repository to your local machine and install
```
git clone git@github.com:nsf-c-cas/rxntorch.git
cd rxntorch
python setup.py install
```

