#!/bin/sh

cd ConvE

# train the original model
echo 'Training original model'

CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data WN18RR --save-influence-map --reproduce-results

echo 'Selecting target triples'
mkdir data/target_conve_WN18RR_0
CUDA_VISIBLE_DEVICES=0 python -u select_targets.py --model conve --data WN18RR  --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u select_rand_targets.py --model conve --data WN18RR  --reproduce-results

echo 'Generating random deletions for the neighbourhood'
CUDA_VISIBLE_DEVICES=0 python -u rand_del_n.py --model conve --data WN18RR
CUDA_VISIBLE_DEVICES=0 python -u rand_del_g.py --model conve --data WN18RR

python -u wrangle_KG.py rand_del_n_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data rand_del_n_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py rand_del_g_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data rand_del_g_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR


echo 'Generating deletions for the neighbourhood using baselines'
CUDA_VISIBLE_DEVICES=0 python -u gr_del.py --model conve --data WN18RR --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u ijcai_del.py --model conve --data WN18RR --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u criage_del.py --model conve --data WN18RR --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u criage_del_2.py --model conve --data WN18RR --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u score_del.py --model conve --data WN18RR --reproduce-results

python -u wrangle_KG.py gr_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data gr_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py ijcai_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data ijcai_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py criage_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data criage_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py criage_del_2_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data criage_del_2_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py score_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data score_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR


echo 'Generating deletions for the neighbourhood using similarity metrics'
CUDA_VISIBLE_DEVICES=0 python -u cos_del.py --model conve --data WN18RR --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u dot_del.py --model conve --data WN18RR --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u l2_del.py --model conve --data WN18RR --reproduce-results

python -u wrangle_KG.py cos_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data cos_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py dot_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data dot_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py l2_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data l2_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR



echo 'Generating deletions for the neighbourhood using gradient based metrics'
CUDA_VISIBLE_DEVICES=0 python -u cos_grad_del.py --model conve --data WN18RR --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u dot_grad_del.py --model conve --data WN18RR --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u l2_grad_del.py --model conve --data WN18RR --reproduce-results

python -u wrangle_KG.py cos_grad_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data cos_grad_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py dot_grad_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data dot_grad_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py l2_grad_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data l2_grad_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR



echo 'Generating deletions for the neighbourhood using influence functions'
CUDA_VISIBLE_DEVICES=0 python -u if_del.py --model conve --data WN18RR --reproduce-results

python -u wrangle_KG.py if_del_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data if_del_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR




echo 'Generating random additions for the neighbourhood'
CUDA_VISIBLE_DEVICES=0 python -u rand_add_n.py --model conve --data WN18RR
CUDA_VISIBLE_DEVICES=0 python -u rand_add_g.py --model conve --data WN18RR

python -u wrangle_KG.py rand_add_n_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data rand_add_n_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py rand_add_g_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data rand_add_g_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR


echo 'Generating additions for the neighbourhood using baselines'
CUDA_VISIBLE_DEVICES=0 python -u ijcai_add.py --model conve --data WN18RR --reproduce-results
python -u wrangle_KG.py ijcai_add_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data ijcai_add_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

CUDA_VISIBLE_DEVICES=0 python -u ijcai_add.py --model conve --data WN18RR --reproduce-results --corruption-factor 20 --budget 2
python -u wrangle_KG.py ijcai_add_conve_WN18RR_0_100_1_2_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data ijcai_add_conve_WN18RR_0_100_1_2_1 --reproduce-results --original-data WN18RR



CUDA_VISIBLE_DEVICES=0 python -u criage_inverter.py --model conve --data WN18RR --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u criage_add.py --model conve --data WN18RR --reproduce-results
python -u wrangle_KG.py criage_add_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data criage_add_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR



echo 'Generating additions for the neighbourhood using similarity metrics'
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model conve --data WN18RR --sim-metric 'cos' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model conve --data WN18RR --sim-metric 'dot' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model conve --data WN18RR --sim-metric 'l2' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model conve --data WN18RR --sim-metric 'score' --reproduce-results

python -u wrangle_KG.py cos_add_5_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data cos_add_5_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py dot_add_5_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data dot_add_5_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py l2_add_5_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data l2_add_5_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py score_add_5_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data score_add_5_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR


echo 'Generating additions for the neighbourhood using similarity of gradients'
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model conve --data WN18RR --sim-metric 'cos_grad' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model conve --data WN18RR --sim-metric 'dot_grad' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model conve --data WN18RR --sim-metric 'l2_grad' --reproduce-results

python -u wrangle_KG.py cos_grad_add_5_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data cos_grad_add_5_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py dot_grad_add_5_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data dot_grad_add_5_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

python -u wrangle_KG.py l2_grad_add_5_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data l2_grad_add_5_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR




echo 'Generating additions for the neighbourhood using influence function'
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model conve --data WN18RR --sim-metric 'if' --reproduce-results
python -u wrangle_KG.py if_add_5_conve_WN18RR_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model conve --data if_add_5_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR












