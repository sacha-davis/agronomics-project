
# Kipoi README

The initial part of this project included the use of the [Kipoi Model Zoo library](http://kipoi.org/) to use the MPRA-DragoNN model. Unfortunately, the use of this library was seemingly incompatible with the most recent versions of Keras and TensorFlow, as well as some other important libraries. 

If you would like to run the notebooks contained in this folder, please use the following commands to set up your environment:
```
conda create --name kipoi_venv python=3.6
conda activate kipoi_venv
pip install --upgrade pip==20.3.3
pip install -r legacy/kipoi/requirements.txt
```

Please note that these notebooks and Kipoi are note compatible with Google Colab at this time.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3Mjg1Nzk1OTYsLTY3MTI0MTg2OCwtNz
k3MDQzMzM3LC0xMDQ2MjI5ODQ4LDMxMjkxNzczMyw3MzA5OTgx
MTZdfQ==
-->