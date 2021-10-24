<h1 align="left">
  Code Structure
</h1>
<h3 align="left">This file describes the structure of the code</h3>

Commandline instructions for all experiments are available in bash scripts at this level
 
The main codebase is in `ConvE`
- script to preprocess data (generate dictionaries) is `preprocess.py`
- script to generate evaluation filters is `wrangle_KG.py`
- script to train a KGE model is `main.py` and the model architecture is in `model.py`
- commmon utilities and hyperparameters are in `utils.py`
- script to select target triples from the test set is `select_targets.py` and to further select random samples `select_rand_targets.py`
- Baseline deletion attacks
    - Random neighbourhood deletions in `rand_del_n.py`
    - Random global deletions in `rand_del_g.py`
    - Direct-Delete attack in `ijcai_del.py`
    - Gradient Rollback based deletions in `gr_del.py`
    - CRIAGE in `criage_del.py`
- Proposed deletion attacks 
    - Instance similarity based attacks in `cos_del.py`, `l2_del.py`, `dot_del.py`
    - Gradient similarity based attacks in `cos_grad_del.py`, `l2_grad_del.py`, `dot_grad_del.py`
    - Influence function in `if_del.py`
- Baseline addition attacks
    - Random neighbourhood additions in `rand_add_n.py`
    - Random global additions in `rand_add_g.py`
    - Direct-Add attack in `ijcai_add.py`
    - CRIAGE attack in `criage_add.py`, inverter model in `criage_inverter.py` and inverter model architecture in `criage_model.py`
- Proposed addition attacks 
    - Dissimilar entity by cosine distance in `if_add_5.py`
    - Dissimilar entity by L2 distance in `if_add_5_l2.py`
- Folder `data` will contain datasets generated from running the experiments. 
    - These are named as `attack_model_dataset_split_budget_run` 
    - here `budget=1`, `run` is the number for a random run and
    - `split=0_100_1` indicates target split where the set of test triples with ranks=1 are targets (`0`), `100` of these are randomly sampled and `1` is the random run.
    - For the 2 versions of Direct-Add attacks (based on 2 different downsampling percents), I used `budget=1` and `budget=2` to differentiate the output files
- Folder `saved_models`, `data`, `logs`, `influence_maps` will contain the outputs if a script is run. Metrics on test set can be checked from `logs`.
