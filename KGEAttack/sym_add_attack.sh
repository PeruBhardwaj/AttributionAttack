#!/bin/sh

cd ConvE

# echo 'Generating symmetry edits with cosine distance : WN18RR DistMult'
# python -u sym_add_attack_3.py --model distmult --data WN18RR --reproduce-results
# python -u wrangle_KG.py sym_add_3_distmult_WN18RR_0_100_1_1_1
# python -u main.py --model distmult --data sym_add_3_distmult_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating symmetry edits with cosine distance : WN18RR ComplEx'
python -u sym_add_attack_3.py --model complex --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_3_complex_WN18RR_0_100_1_1_1
python -u main.py --model complex --data sym_add_3_complex_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating symmetry edits with cosine distance : WN18RR ConvE'
python -u sym_add_attack_3.py --model conve --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_3_conve_WN18RR_0_100_1_1_1
python -u main.py --model conve --data sym_add_3_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating symmetry edits with cosine distance : WN18RR Transe'
python -u sym_add_attack_3.py --model transe --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_3_transe_WN18RR_0_100_1_1_1
python -u main.py --model transe --data sym_add_3_transe_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

####################################################################################################################################

echo 'Generating symmetry edits with cosine distance : FB15k-237 DistMult'
python -u sym_add_attack_3.py --model distmult --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_3_distmult_FB15k-237_0_100_1_1_1
python -u main.py --model distmult --data sym_add_3_distmult_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating symmetry edits with cosine distance : FB15k-237 ComplEx'
python -u sym_add_attack_3.py --model complex --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_3_complex_FB15k-237_0_100_1_1_1
python -u main.py --model complex --data sym_add_3_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating symmetry edits with cosine distance : FB15k-237 ConvE'
python -u sym_add_attack_3.py --model conve --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_3_conve_FB15k-237_0_100_1_1_1
python -u main.py --model conve --data sym_add_3_conve_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating symmetry edits with cosine distance : FB15k-237 Transe'
python -u sym_add_attack_3.py --model transe --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_3_transe_FB15k-237_0_100_1_1_1
python -u main.py --model transe --data sym_add_3_transe_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

###################################################################################################################################
###################################################################################################################################

echo 'Generating symmetry edits with worse ranks : WN18RR DistMult'
python -u sym_add_attack_2.py --model distmult --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_2_distmult_WN18RR_0_100_1_1_1
python -u main.py --model distmult --data sym_add_2_distmult_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating symmetry edits with worse ranks : WN18RR ComplEx'
python -u sym_add_attack_2.py --model complex --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_2_complex_WN18RR_0_100_1_1_1
python -u main.py --model complex --data sym_add_2_complex_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating symmetry edits with worse ranks : WN18RR ConvE'
python -u sym_add_attack_2.py --model conve --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_2_conve_WN18RR_0_100_1_1_1
python -u main.py --model conve --data sym_add_2_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating symmetry edits with worse ranks : WN18RR Transe'
python -u sym_add_attack_2.py --model transe --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_2_transe_WN18RR_0_100_1_1_1
python -u main.py --model transe --data sym_add_2_transe_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

####################################################################################################################################

echo 'Generating symmetry edits with worse ranks : FB15k-237 DistMult'
python -u sym_add_attack_2.py --model distmult --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_2_distmult_FB15k-237_0_100_1_1_1
python -u main.py --model distmult --data sym_add_2_distmult_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating symmetry edits with worse ranks : FB15k-237 ComplEx'
python -u sym_add_attack_2.py --model complex --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_2_complex_FB15k-237_0_100_1_1_1
python -u main.py --model complex --data sym_add_2_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating symmetry edits with worse ranks : FB15k-237 ConvE'
python -u sym_add_attack_2.py --model conve --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_2_conve_FB15k-237_0_100_1_1_1
python -u main.py --model conve --data sym_add_2_conve_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating symmetry edits with worse ranks : FB15k-237 Transe'
python -u sym_add_attack_2.py --model transe --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_2_transe_FB15k-237_0_100_1_1_1
python -u main.py --model transe --data sym_add_2_transe_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

###################################################################################################################################
###################################################################################################################################

echo 'Generating symmetry edits with ground truth : WN18RR DistMult'
python -u sym_add_attack_1.py --model distmult --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_1_distmult_WN18RR_0_100_1_1_1
python -u main.py --model distmult --data sym_add_1_distmult_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating symmetry edits with ground truth : WN18RR ComplEx'
python -u sym_add_attack_1.py --model complex --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_1_complex_WN18RR_0_100_1_1_1
python -u main.py --model complex --data sym_add_1_complex_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating symmetry edits with ground truth : WN18RR ConvE'
python -u sym_add_attack_1.py --model conve --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_1_conve_WN18RR_0_100_1_1_1
python -u main.py --model conve --data sym_add_1_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating symmetry edits with ground truth : WN18RR Transe'
python -u sym_add_attack_1.py --model transe --data WN18RR --reproduce-results
python -u wrangle_KG.py sym_add_1_transe_WN18RR_0_100_1_1_1
python -u main.py --model transe --data sym_add_1_transe_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

# ####################################################################################################################################

echo 'Generating symmetry edits with ground truth : FB15k-237 DistMult'
python -u sym_add_attack_1.py --model distmult --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_1_distmult_FB15k-237_0_100_1_1_1
python -u main.py --model distmult --data sym_add_1_distmult_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating symmetry edits with ground truth : FB15k-237 ComplEx'
python -u sym_add_attack_1.py --model complex --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_1_complex_FB15k-237_0_100_1_1_1
python -u main.py --model complex --data sym_add_1_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating symmetry edits with ground truth : FB15k-237 ConvE'
python -u sym_add_attack_1.py --model conve --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_1_conve_FB15k-237_0_100_1_1_1
python -u main.py --model conve --data sym_add_1_conve_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating symmetry edits with ground truth : FB15k-237 Transe'
python -u sym_add_attack_1.py --model transe --data FB15k-237 --reproduce-results
python -u wrangle_KG.py sym_add_1_transe_FB15k-237_0_100_1_1_1
python -u main.py --model transe --data sym_add_1_transe_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

