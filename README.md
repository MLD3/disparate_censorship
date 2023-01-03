# Disparate Censorship: A Mechanism for Model Performance Gaps in Clinical Machine Learning

Official code repository for the paper "[Disparate Censorship: A Mechanism for Model Performance Gaps in Clinical Machine Learning](https://arxiv.org/abs/2208.01127)" (MLHC 2022).

## Experiments

All scripts for experiments are in the `scripts/` directory.

* Setting 2: Run the scripts under `scripts/` in _this_ directory (i.e. `bash scripts/setting2_full.csv`). 
    - `scripts/setting2_full.csv`: Performance gap results for models at varying testing thresholds. See `simulation_results.ipynb` to access/plot results.
    - `scripts/setting2_variance.csv`, `scripts/setting2_mean_distance.csv`: Supplementary results under different distributional parameters at a fixed testing threshold. See `simulation_results.ipynb` to access/plot results.
* Setting 3: Run the Python script `script/conditional_shift_experiments.py`. See `simulation_results.ipynb` to access/plot results.

Raw results will be outputted into the `results/` directory, where the results for each individual run (i.e. threshold/distribution parameter setting) and associated replications are stored in CSV files (e.g. `results/SETTING2_BASE_5_5/*.csv`). See `simulation_results.ipynb` for an example of how to read the raw results.

## Figures

Plots can be reconstructed using each of the notebooks:
* `simulation_examples.ipynb`: 2D toy examples of simulated disparate censorship
* `simulation_results.ipynb`: Plots for performance gaps in Setting 2 and 3 of the paper
* `mimic_results.ipynb`: Plot (singular) and hypothesis test results for all MIMIC-IV v1.0 results

All figures will be outputted into the `figures/` directory.

## Citation

If you use this repository or our paper for your own research, we'd love for you to cite our work:
```
@inproceedings{chang2022disparate,
  title={Disparate Censorship: A Mechanism for Model Performance Gaps in Clinical Machine Learning},
  author={Chang, Trenton and Sjoding, Michael W and Wiens, Jenna},
  booktitle={Machine Learning for Healthcare Conference},
  volume={182},
  year={2022},
  organization={PMLR}
}

```
