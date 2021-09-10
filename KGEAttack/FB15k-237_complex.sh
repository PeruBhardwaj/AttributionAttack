#!/bin/sh

cd ConvE


# # train the original model
echo 'Training original model'

CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data FB15k-237 --save-influence-map --reproduce-results

echo 'Selecting target triples'
mkdir data/target_complex_FB15k-237_0
CUDA_VISIBLE_DEVICES=0 python -u select_targets.py --model complex --data FB15k-237  --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u select_rand_targets.py --model complex --data FB15k-237 --reproduce-results

echo 'Generating random deletions for the neighbourhood'
CUDA_VISIBLE_DEVICES=0 python -u rand_del_n.py --model complex --data FB15k-237
CUDA_VISIBLE_DEVICES=0 python -u rand_del_g.py --model complex --data FB15k-237

python -u wrangle_KG.py rand_del_n_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data rand_del_n_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py rand_del_g_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data rand_del_g_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237


echo 'Generating deletions for the neighbourhood using baselines'
CUDA_VISIBLE_DEVICES=0 python -u gr_del.py --model complex --data FB15k-237 --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u ijcai_del.py --model complex --data FB15k-237 --reproduce-results
# CUDA_VISIBLE_DEVICES=0 python -u criage_del.py --model complex --data FB15k-237 --reproduce-results
# CUDA_VISIBLE_DEVICES=0 python -u criage_del_2.py --model complex --data FB15k-237 --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u score_del.py --model complex --data FB15k-237 --reproduce-results

python -u wrangle_KG.py gr_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data gr_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py ijcai_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data ijcai_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

# python -u wrangle_KG.py criage_del_complex_FB15k-237_0_100_1_1_1
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data criage_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

# python -u wrangle_KG.py criage_del_2_complex_FB15k-237_0_100_1_1_1
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data criage_del_2_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py score_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data score_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237


echo 'Generating deletions for the neighbourhood using similarity metrics'
CUDA_VISIBLE_DEVICES=0 python -u cos_del.py --model complex --data FB15k-237 --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u dot_del.py --model complex --data FB15k-237 --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u l2_del.py --model complex --data FB15k-237 --reproduce-results

python -u wrangle_KG.py cos_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data cos_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py dot_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data dot_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py l2_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data l2_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237



echo 'Generating deletions for the neighbourhood using gradient based metrics'
CUDA_VISIBLE_DEVICES=0 python -u cos_grad_del.py --model complex --data FB15k-237 --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u dot_grad_del.py --model complex --data FB15k-237 --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u l2_grad_del.py --model complex --data FB15k-237 --reproduce-results

python -u wrangle_KG.py cos_grad_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data cos_grad_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py dot_grad_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data dot_grad_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py l2_grad_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data l2_grad_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237



echo 'Generating deletions for the neighbourhood using influence functions'
CUDA_VISIBLE_DEVICES=0 python -u if_del.py --model complex --data FB15k-237 --reproduce-results

python -u wrangle_KG.py if_del_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data if_del_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237


echo 'Generating random additions for the neighbourhood'
CUDA_VISIBLE_DEVICES=0 python -u rand_add_g.py --model complex --data FB15k-237
CUDA_VISIBLE_DEVICES=0 python -u rand_add_n.py --model complex --data FB15k-237

python -u wrangle_KG.py rand_add_n_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data rand_add_n_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py rand_add_g_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data rand_add_g_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237


echo 'Generating additions for the neighbourhood using baselines'
CUDA_VISIBLE_DEVICES=0 python -u ijcai_add.py --model complex --data FB15k-237 --reproduce-results
python -u wrangle_KG.py ijcai_add_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data ijcai_add_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

CUDA_VISIBLE_DEVICES=0 python -u ijcai_add.py --model complex --data FB15k-237 --reproduce-results --corruption-factor 20 --budget 2
python -u wrangle_KG.py ijcai_add_complex_FB15k-237_0_100_1_2_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data ijcai_add_complex_FB15k-237_0_100_1_2_1 --reproduce-results --original-data FB15k-237



echo 'Generating additions for the neighbourhood using similarity metrics'
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model complex --data FB15k-237 --sim-metric 'cos' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model complex --data FB15k-237 --sim-metric 'dot' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model complex --data FB15k-237 --sim-metric 'l2' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model complex --data FB15k-237 --sim-metric 'score' --reproduce-results

python -u wrangle_KG.py cos_add_5_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data cos_add_5_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py dot_add_5_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data dot_add_5_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py l2_add_5_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data l2_add_5_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py score_add_5_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data score_add_5_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237



echo 'Generating additions for the neighbourhood using similarity of gradients'
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model complex --data FB15k-237 --sim-metric 'cos_grad' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model complex --data FB15k-237 --sim-metric 'dot_grad' --reproduce-results
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model complex --data FB15k-237 --sim-metric 'l2_grad' --reproduce-results

python -u wrangle_KG.py cos_grad_add_5_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data cos_grad_add_5_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py dot_grad_add_5_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data dot_grad_add_5_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

python -u wrangle_KG.py l2_grad_add_5_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data l2_grad_add_5_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237


echo 'Generating additions for the neighbourhood using influence function'
CUDA_VISIBLE_DEVICES=0 python -u if_add_5.py --model complex --data FB15k-237 --sim-metric 'if' --reproduce-results
python -u wrangle_KG.py if_add_5_complex_FB15k-237_0_100_1_1_1
CUDA_VISIBLE_DEVICES=0 python -u main.py --model complex --data if_add_5_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237


















