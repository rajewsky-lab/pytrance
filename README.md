# pyTrance

A **Py**thon framework for **tran**script **c**o-localization analysis based on latent **e**mbeddings for imaging-based spatial transcriptomics data.  

Have a look at our preprint and documentation for more details.

## Installation
First, download the code to your desired location in your filesystem:
```
git clone https://github.com/rajewsky-lab/pyTrance/
cd pyTrance
```
To install the pyTrance dependencies we highly recommend using [mamba](https://github.com/mamba-org/mamba) as a fast, drop-in replacement for `conda`. Alternatively, replace `mamba` with `conda` in the command bellow.:
```
mamba env create -f environment.yaml
conda activate pytrance
```

For the installation of PyTorch please follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to make sure it is compatible with the CUDA version on your machine.

Finally, install pyTrance:
```
pip install pytrance
```