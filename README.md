# Personalization using enrollment

This repository contains the code needed to replicate the experiments presented in the accompanied paper.

## List of python files

1. `main.py`: code used to start training, utilizing `hydra` configuration files in `configs`
2. `models.py`: code to create models
3. `data.py`: implementation of datasets
4. `evaluate.py`: code to compute Gini index and CIs
5. `iswf.py`: code to create ISWF plots
6. `predict_test_msp.py`: code to evalute on MSP test set
7. `speaker_plot.py`: code to plot speaker-level UAR (not used in paper)
8. `training.py`: code implementing training

## Adaptation data
Adaptation CSVs are included under `adaptation-sets` for each dataset/task.
Only filenames are included (need to request datasets from respective owners).