This repository contains the python code to implement [spectral graph model (SGM)](https://www.sciencedirect.com/science/article/pii/S1053811922000490) and fit it via [simulation-based inference (SBI)](https://www.pnas.org/doi/abs/10.1073/pnas.1912789117). 


SGM is a neuro-model which can capture the frequency power spectra (PSD) and functional connectivity (FC) obtained from MEG recordings.
Although not fully confirmed, SGM may also be applicable to EEG data.
This repository also includes a version of the SGM implementation specifically designed for fMRI data (`sgm_fMRI`).

For the detailed usage of our SGM and the code, please refer to `notebooks` folder which includes

- `SGM_tutorial.ipynb`: How to use SGM model to simulate PSD or FC given a set of SGM parameters (for MEG or EEG).  

- `SBI_SGM_tutorial.ipynb`: How to use SBI to fit SGM model with fitting target FC, PSD or both (for MEG or EEG).

- `sgmfMRI_tutorial.ipynb`: How to use optimization or annealing to fit sgmfMRI with given fMRI recording. 

While the this repository contains demo datasets for the tutorials, 
to run the code on your own dataset, you should prepare

- MEG/EEG recordings or PSD or FC (or fMRI recording)

- The corresponding structrual connectivity (SC) matrix. 

- The corresponding distance matrix (`sgmfMRI` not require).



To run it, it is suggested to use docker with image `huaqingjin/sgm:mne_lib_fc`. 

If you find the image too complex, ensure your Python environment includes the following packages:

- sbi
- mne
- torch
- easydict

Additionally, it should also include other standard packages found in Anaconda.


If you find this repository useful, please cite 

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
