# NRC / Amii Agronomics Project

## Introduction
Recent changes in global climate patterns (such as extreme average temperatures, droughts, and floods) have made the yields of Canadian farming methods much more unpredictable. Given these trends, scientists have taken to examining another major factor that can affect the success of a growing season: the genetics of the crop organism.

In the domain of genetics, certain segments of DNA can act as gene expression enhancers, giving rise to an increase in the number of RNA transcripts from that region which are eventually translated into proteins. Proteins are the building blocks of life and can have a significant effect on the cellular system they reside in, which can, in turn, lead to interesting downstream phenotypes in the host organism. Gene expression prediction is of particular interest to those in domains such as genetic engineering, which continues to be a key player in precision agriculture and smart food production systems.

The primary goal of this project is to train a machine learning model that can accurately predict an increase or decrease in gene expression caused by the presence of a particular DNA sequence within Canadian protein crops. In this pursuit, we test a number of combiantions of interesting deep learning architectures with different genetic data sources.

## Notes
As an early starting point, the [Kipoi Model Zoo](http://kipoi.org/) library was used to access the MPRA-DragoNN model. The environment required to run Kipoi is not compatible with the environment used for the rest of the project, so if you would like to run these files, please refer to the README in `legacy/kipoi`.

This repository can be run using Jupyter notebook + conda for a local machine, or on Google Drive using Google Colab.

## File Naming Conventions
There are two types of executable files in this repository, `.py` and `.ipynb`. Each shares a similar naming structure. Each of these filenames consists of multiple parts:

`<purpose_code>_<organism>-<title>-<version>-<qualifier>.<extension>`

1. `<purpose_code>` denotes the function of the file, and can be either DAU (Data Acquisition and Understanding), MLME (Machine Learning & Evaluation), or both.
2. `<organism>` refers to the dataset this file applies to, which is either canola or arabidopsis. Files that do not contain this are general.
3. `<title>` (also knows as the notebook name) briefly describes what the functionality of the file is. This is maintained between versions of the file.
4. `<version>` works in conjunction with `<title>` to show the progression of the file over time.
6. `<qualifier>` adds more detail when there are multiple versions of a file.
7. `<extension>` is either `.py` or `.ipynb` to show if the file is a notebook or a script.

## Running Instructions
0. **Clone repository and nagivate to the root directory.** 
1. **If using Arabidopsis data, copy the data file into the data folder.**
		```
	cp <path to local dataset> data/raw/athal_istarr_hidra.tsv
		```
2. **Set up environment and initialize notebook**.
	1.1. If using conda and Jupyter, in the shell:
	```
	conda create --name agronomics_venv python=3.8
	conda activate agronomics_venv
	pip install -r requirements.txt
	jupyter notebook
	```
	1.2. If using GDrive and Colab, open `driver.ipynb` .


3.  **Run Code**.  
	2.1.  To run without changing any parameters, only run code located in the `Run Code` section. 
	2.2.  To change default parameters or even perform a gridsearch, please refer to the `Running Examples` section for guides. The arguments for these files are specified in each file's `fetch_args` function.

## Credits
`Movva R, Greenside P, Marinov GK, Nair S, Shrikumar A, Kundaje A (2019). Deciphering regulatory DNA sequences and noncoding genetic variants using neural network models of massively parallel reporter assays. PLoS ONE 14(6): e0218073.  [https://doi.org/10.1371/journal.pone.0218073](https://doi.org/10.1371/journal.pone.0218073)`

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzMzUwMjg5NjEsMTQzNDc2OTA4NCwtMT
E2NzQ1NjU2NCwyMTM4NTI2Mzg5LC04NzIyOTczNjksLTExNDY3
NDIyNTcsLTEwMTA5NzUwNzQsLTc4ODkwODI4OCwtNTQyNTEzOT
MwLDEyNDQwNDk0NTQsMTc0MTMxNTE3MCwxODAwMzA3NDAwLDcz
MDk5ODExNl19
-->