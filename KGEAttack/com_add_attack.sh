#!/bin/sh

cd ConvE

echo 'Generating composition edits with cosine distance : WN18RR DistMult'
python -u com_add_attack_3.py --model distmult --data WN18RR --reproduce-results
python -u wrangle_KG.py com_add_3_distmult_WN18RR_0_100_1_1_1
python -u main.py --model distmult --data com_add_3_distmult_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating composition edits with cosine distance : WN18RR ComplEx'
python -u com_add_attack_3.py --model complex --data WN18RR --reproduce-results
python -u wrangle_KG.py com_add_3_complex_WN18RR_0_100_1_1_1
python -u main.py --model complex --data com_add_3_complex_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating composition edits with cosine distance : WN18RR ConvE'
python -u com_add_attack_3.py --model conve --data WN18RR --reproduce-results
python -u wrangle_KG.py com_add_3_conve_WN18RR_0_100_1_1_1
python -u main.py --model conve --data com_add_3_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating composition edits with cosine distance : WN18RR Transe'
python -u com_add_attack_3.py --model transe --data WN18RR --reproduce-results
python -u wrangle_KG.py com_add_3_transe_WN18RR_0_100_1_1_1
python -u main.py --model transe --data com_add_3_transe_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

####################################################################################################################################

echo 'Generating composition edits with cosine distance : FB15k-237 DistMult'
python -u com_add_attack_3.py --model distmult --data FB15k-237 --reproduce-results
python -u wrangle_KG.py com_add_3_distmult_FB15k-237_0_100_1_1_1
python -u main.py --model distmult --data com_add_3_distmult_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating composition edits with cosine distance : FB15k-237 ComplEx'
python -u com_add_attack_3.py --model complex --data FB15k-237 --reproduce-results
python -u wrangle_KG.py com_add_3_complex_FB15k-237_0_100_1_1_1
python -u main.py --model complex --data com_add_3_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating composition edits with cosine distance : FB15k-237 ConvE'
python -u com_add_attack_3.py --model conve --data FB15k-237 --reproduce-results
python -u wrangle_KG.py com_add_3_conve_FB15k-237_0_100_1_1_1
python -u main.py --model conve --data com_add_3_conve_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating composition edits with cosine distance : FB15k-237 Transe'
python -u com_add_attack_3.py --model transe --data FB15k-237 --reproduce-results
python -u wrangle_KG.py com_add_3_transe_FB15k-237_0_100_1_1_1
python -u main.py --model transe --data com_add_3_transe_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

###################################################################################################################################
###################################################################################################################################

echo 'Generating composition edits with worse ranks : WN18RR DistMult'
python -u com_add_attack_2.py --model distmult --data WN18RR --reproduce-results
python -u wrangle_KG.py com_add_2_distmult_WN18RR_0_100_1_1_1
python -u main.py --model distmult --data com_add_2_distmult_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating composition edits with worse ranks : WN18RR ComplEx'
python -u com_add_attack_2.py --model complex --data WN18RR --reproduce-results
python -u wrangle_KG.py com_add_2_complex_WN18RR_0_100_1_1_1
python -u main.py --model complex --data com_add_2_complex_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating composition edits with worse ranks : WN18RR ConvE'
python -u com_add_attack_2.py --model conve --data WN18RR --reproduce-results
python -u wrangle_KG.py com_add_2_conve_WN18RR_0_100_1_1_1
python -u main.py --model conve --data com_add_2_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating composition edits with worse ranks : WN18RR Transe'
python -u com_add_attack_2.py --model transe --data WN18RR --reproduce-results
python -u wrangle_KG.py com_add_2_transe_WN18RR_0_100_1_1_1
python -u main.py --model transe --data com_add_2_transe_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

###################################################################################################################################

echo 'Generating composition edits with worse ranks : FB15k-237 DistMult'
python -u com_add_attack_2.py --model distmult --data FB15k-237 --reproduce-results
python -u wrangle_KG.py com_add_2_distmult_FB15k-237_0_100_1_1_1
python -u main.py --model distmult --data com_add_2_distmult_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating composition edits with worse ranks : FB15k-237 ComplEx'
python -u com_add_attack_2.py --model complex --data FB15k-237 --reproduce-results
python -u wrangle_KG.py com_add_2_complex_FB15k-237_0_100_1_1_1
python -u main.py --model complex --data com_add_2_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating composition edits with worse ranks : FB15k-237 ConvE'
python -u com_add_attack_2.py --model conve --data FB15k-237 --reproduce-results
python -u wrangle_KG.py com_add_2_conve_FB15k-237_0_100_1_1_1
python -u main.py --model conve --data com_add_2_conve_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating composition edits with worse ranks : FB15k-237 Transe'
python -u com_add_attack_2.py --model transe --data FB15k-237 --reproduce-results
python -u wrangle_KG.py com_add_2_transe_FB15k-237_0_100_1_1_1
python -u main.py --model transe --data com_add_2_transe_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

##################################################################################################################################
##################################################################################################################################

echo 'Generating composition edits with ground truth : WN18RR DistMult'
python -u create_clusters.py --model distmult --data WN18RR --num-clusters 300 --reproduce-results
python -u com_add_attack_1.py --model distmult --data WN18RR --reproduce-results --num-clusters 300
python -u wrangle_KG.py com_add_1_distmult_WN18RR_0_100_1_1_1
python -u main.py --model distmult --data com_add_1_distmult_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating composition edits with ground truth : WN18RR ComplEx'
python -u create_clusters.py --model complex --data WN18RR --num-clusters 300 --reproduce-results
python -u com_add_attack_1.py --model complex --data WN18RR --reproduce-results --num-clusters 300
python -u wrangle_KG.py com_add_1_complex_WN18RR_0_100_1_1_1
python -u main.py --model complex --data com_add_1_complex_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating composition edits with ground truth : WN18RR ConvE'
python -u create_clusters.py --model conve --data WN18RR --num-clusters 300 --reproduce-results
python -u com_add_attack_1.py --model conve --data WN18RR --reproduce-results --num-clusters 300
python -u wrangle_KG.py com_add_1_conve_WN18RR_0_100_1_1_1
python -u main.py --model conve --data com_add_1_conve_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

echo 'Generating composition edits with ground truth : WN18RR Transe'
python -u create_clusters.py --model transe --data WN18RR --num-clusters 100 --reproduce-results
python -u com_add_attack_1.py --model transe --data WN18RR --reproduce-results --num-clusters 100
python -u wrangle_KG.py com_add_1_transe_WN18RR_0_100_1_1_1
python -u main.py --model transe --data com_add_1_transe_WN18RR_0_100_1_1_1 --reproduce-results --original-data WN18RR

####################################################################################################################################

echo 'Generating composition edits with ground truth : FB15k-237 DistMult'
python -u create_clusters.py --model distmult --data FB15k-237 --num-clusters 300 --reproduce-results
python -u com_add_attack_1.py --model distmult --data FB15k-237 --reproduce-results --num-clusters 300
python -u wrangle_KG.py com_add_1_distmult_FB15k-237_0_100_1_1_1
python -u main.py --model distmult --data com_add_1_distmult_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating composition edits with ground truth : FB15k-237 ComplEx'
python -u create_clusters.py --model complex --data FB15k-237 --num-clusters 300 --reproduce-results
python -u com_add_attack_1.py --model complex --data FB15k-237 --reproduce-results --num-clusters 300
python -u wrangle_KG.py com_add_1_complex_FB15k-237_0_100_1_1_1
python -u main.py --model complex --data com_add_1_complex_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating composition edits with ground truth : FB15k-237 ConvE'
python -u create_clusters.py --model conve --data FB15k-237 --num-clusters 300 --reproduce-results
python -u com_add_attack_1.py --model conve --data FB15k-237 --reproduce-results --num-clusters 300
python -u wrangle_KG.py com_add_1_conve_FB15k-237_0_100_1_1_1
python -u main.py --model conve --data com_add_1_conve_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

echo 'Generating composition edits with ground truth : FB15k-237 Transe'
python -u create_clusters.py --model transe --data FB15k-237 --num-clusters 100 --reproduce-results
python -u com_add_attack_1.py --model transe --data FB15k-237 --reproduce-results --num-clusters 100
python -u wrangle_KG.py com_add_1_transe_FB15k-237_0_100_1_1_1
python -u main.py --model transe --data com_add_1_transe_FB15k-237_0_100_1_1_1 --reproduce-results --original-data FB15k-237

