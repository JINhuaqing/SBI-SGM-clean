This repository is the clean version code of SBI-SGM

Plz refer to `notebooks/SBI_SGM_tutorial.ipynb` for the usage

To run it, it is suggested to use docker with image `huaqingjin/sgm:umap`

**In current version, we fit SGM model with PSD features only(not including spatial features)**

To run our method, you may need 

1. Your own psd data with sc matrix and dist matrix in DK atlas (nroi=68)


If use it, plz cite my paper

```
@article{jin2023bayesian,
title={Bayesian inference of a spectral graph model for brain oscillations},
author={Jin, Huaqing and Verma, Parul and Jiang, Fei and Nagarajan, Srikantan S and Raj, Ashish},
journal={NeuroImage},
volume={279},
pages={120278},
year={2023},
publisher={Elsevier}
}
```
