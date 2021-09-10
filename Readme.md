### Adversarial Attacks on Knowledge Graph Embeddings via Instance Attribution Methods
This is the code repository to accompany EMNLP-2021 paper.

Below we describe the steps to reproduce the results for the experiments.


### Dependencies
- python = 3.8.5
- pytorch = 1.4.0
- numpy = 1.19.1
- jupyter = 1.0.0
- pandas = 1.1.0
- matplotlib = 3.2.2
- scikit-learn = 0.23.2
- seaborn = 0.11.0

We have also included the conda environment file attribution_attack.yml


### Reproducing the results
- To preprocess the original dataset, use the bash script preprocess.sh
- For each model-dataset combination, we have included a bash script to train the original model, generate attacks from baselines and proposed attacks; and train poisoned model. These scripts are named as model-dataset.sh
- The instructions in these scripts are grouped together under the echo statements which indicate what they do.
- The commandline argument --reproduce-results uses the hyperparameters that were used for the experiments reported in the submission. These hyperparameter values can be inspected in the function set_hyperparams() in utils.py
- To reproduce the results, specific instructions from the bash scripts can be run on commandline or the full script can be run




