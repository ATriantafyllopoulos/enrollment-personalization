# Personalization using enrollment

This repository contains the code needed to reproduce the experiments in (Triantafyllopoulos and Schuller, 2024).

![Method Overview][./assets/Personalization.png]

## List of python files

1. `main.py`: code used to start training, utilizing `hydra` configuration files in `configs`
2. `models.py`: code to create models
3. `data.py`: implementation of datasets
4. `evaluate.py`: code to compute Gini index and CIs
5. `iswf.py`: code to create ISWF plots
6. `predict_test_msp.py`: code to evalute on MSP test set
7. `speaker_plot.py`: code to plot speaker-level UAR (not used in paper)
8. `training.py`: code implementing training

## Usage
1. Download MSP-Podcast and FAU-AIBO manually. To do this, you need an EULA with the dataset owners.
2. Run `main.py`. You will be asked to provide the `root` for the data and the `results-root` to store your results.
3. (Optional) Run `evaluate.py` and `iswf.py` to run the detailed evaluations of the paper.

## Adaptation data
Adaptation CSVs are included under `adaptation-sets` for each dataset/task.
Only filenames are included (need to request datasets from respective owners).

## Reference
```
Triantafyllopoulos, A., Schuller, B., (2024), "Enrolment-based personalisation for improving individual-level fairness in speech emotion recognition," Proc. INTERSPEECH, Kos Island, Greece, (accepted).
```

```
@inproceedings{Triantafyllopoulos24-EPF
    author={Triantafyllopoulos, Andreas, and Schuller, Björn},
    title={Enrolment-based personalisation for improving individual-level fairness in speech emotion recognition},
    year={2024},
    booktitle={Proc. INTERSPEECH},
    address={Kos Island, Greece}
}
```