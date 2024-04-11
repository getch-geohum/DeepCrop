# DeepCrop
This repository is mainly dedicated for the integration of multi-source multitemporal earth observation data for crop type mapping using machine and deep learning models designed to handle time series data. The experiment is done in data scarce smallholder farming areas. In addition to mapping crop types, the scripts are used for crop yield modeling from time series vegetation indices. 

Data access, preprocessing, sample preparation and visualization scripts are included inside notebooks folder


this repository also uses core implementations from [Rußwurm et al (2020)]( https://github.com/dl4sits/breizhcrops).


The study hs used Sentinel-2 from ESA, PlanetScope provided as [Norway's International Climate and Forests Initiative Satellite  Data Program(NICFI)](https://www.planet.com/nicfi/), SkySAT from planet labs and time series TerraSAR-X datasets. Please note that SkySAT dataset is not time series and TerraSAR-X dataset has 7 sceneces. Sentinel-2 and PlanetScope monthly composit data can be downloaded freely using [data access and visualization notebook](https://github.com/getch-geohum/DeepCrop/blob/master/notebooks/data_access_preprocess.ipynb). PanetScope monthly composites can alos be explored as base maps from [Planet Exlporer](www.planet.com/explorer). High resolution TerraSAR-X and SkySAT can be accessed upon request given we don't have licence to share.
