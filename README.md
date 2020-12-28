# 🏔 ALASKA

[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)](https://github.com/YassineYousfi/alaska/issues)
[![Generic badge](https://img.shields.io/badge/Status-Beta-ffa500.svg)](https://github.com/YassineYousfi/alaska/pulse)

This repo provides state-of-the-art pre-trained models for steganalysis in the JPEG domain, trained and used to win the [ALASKA 1 steganalaysis challenge](https://alaska.utt.fr/#HallOfFame). Details about the architectures can be found in our [paper](http://www.ws.binghamton.edu/fridrich/Research/ALASKA-preprint1.pdf).

## Important Update
A new Alaska competition is now running on [Kaggle](https://www.kaggle.com/c/alaska2-image-steganalysis/overview), note that the settings are very different from the first edition of the competition: Image sizes, Quality factors, Embedding schemes, and Payload. You can still use the pre-trained models provided for download.
We have open-sourced our solution on [this repo](https://github.com/BloodAxe/Kaggle-2020-Alaska2).

## Features

- Color seperated feature maps extraction using pretrained [SRNet](http://www.ws.binghamton.edu/fridrich/Research/SRNet.pdf) models
- Arbitrary size steganalysis using pretrained detectors 
- Notebooks to fine-tune feature extractors and train custom detectors
- Models are shared within the Tensorflow framework, and converted to [ONNX](https://github.com/onnx/onnx) for use with other deep learning frameworks.

Please note that shared models are only for JPEG quality factor 95.

## Dependecies

Python 3.5+ and dependencies listed in `requirements.txt`.
A Python3 compatible [jpeg Package](http://www.ifs.schaathun.net/pysteg/pysteg.jpeg.html) is included in the tools folder.
System requirements: Mac OS, Linux (tested on Ubuntu 18.04)

## Getting started - Downloading models

Please run the following python code to download the available models.

```python
import requests
import zipfile
import os
home = os.path.expanduser("~")
user = home.split('/')[-1]

url = 'http://dde.binghamton.edu/download/alaska/models.zip'
local = home + '/alaska/models.zip'

r = requests.get(url)
with open(local, 'wb') as f:
    for chunk in tqdm(r.iter_content(chunk_size=2**10)): 
        if chunk:
            f.write(chunk)
with zipfile.ZipFile(local, 'r') as zipref:
    zipref.extractall(home + '/alaska/')
    
os.remove(local)
```

## Getting started - Downloading datasets

This repo comes with minimal image examples, the complete datasets used to train these models have been removed from the official Alaska website by the organizers.

## References

Please consider citing our paper if you find this repository useful.

```
@inproceedings{Yousfi2019Alaska,
 author = {Yousfi, Yassine and Butora, Jan and Fridrich, Jessica and Giboulot, Quentin},
 title = {Breaking ALASKA: Color Separation for Steganalysis in JPEG Domain},
 booktitle = {Proceedings of the ACM Workshop on Information Hiding and Multimedia Security},
 series = {IH\&\#38;MMSec'19},
 year = {2019},
 isbn = {978-1-4503-6821-6},
 location = {Paris, France},
 pages = {138--149},
 numpages = {12},
 url = {http://doi.acm.org/10.1145/3335203.3335727},
 doi = {10.1145/3335203.3335727},
 acmid = {3335727},
 publisher = {ACM},
 address = {New York, NY, USA},
} 
```
