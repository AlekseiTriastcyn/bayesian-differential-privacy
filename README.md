# Bayesian Differential Privacy
Code for the paper "Bayesian Differential Privacy for Machine Learning" (https://arxiv.org/pdf/1901.09697.pdf).

The main code for Bayesian accounant is located in ``bayesian_privacy_accounant.py``.
File ``scaled_renyi.py`` contains a function to compute scaled Renyi divergence of two Gaussian distributions with equal variances. In a similar fashion, functions for other distributions can be added to accomodate other privacy mechanisms.
IPython notebooks implement experiments from the paper.

# Citation
Please cite our paper if find the code helpful:

```
@inproceedings{triastcyn2020bayesian,
  author    = {Triastcyn, Aleksei and Faltings, Boi},
  title     = {Bayesian Differential Privacy for Machine Learning},
  booktitle = {International Conference on Machine Learning},
  year      = {2020}
}
```
