#!/usr/bin/env python
# coding: utf-8

# In this notebook, I add a triple in the neighbourhood of the target triple based on the **IJCAI additions scores** 
# 
# - neighbourhood refers to the triples that share the entities with target's entities
# - I get the addition from both neighbourhood of s and o, then choose the one with higher score
# 
# 

# In[1]:


import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import operator

import json
import logging
import argparse 
import math
from pprint import pprint
import errno

import time

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch.autograd as autograd

from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe
import utils


def get_nghbr_s_addition(test_trip, model, ents, rels, num_cor, epsilon, lambda2, lambda3):
    # get candidates for addition to neighbourhood of s
    # i.e. get candidates for o' and r'
    cand_rd = rng.choice(a=rels, size = num_cor, replace=True) # this is (num_cor,)
    cand_od = rng.choice(a=ents, size = num_cor, replace=True) # this is (num_cor,)
    
#     sub = test_trip[0]
#     nghbr_mask = (np.isin(per_tr[:,0], [sub]) | np.isin(per_tr[:,2], [sub]))
#     test_neighbours = np.where(nghbr_mask)[0] # this is index of neighbours in training data
#     nghbr_trip = per_tr[test_neighbours] # actual neighbour triples
    
    # --- Get the perturbed subject embedding ---
    test_trip = torch.from_numpy(test_trip).to(device)[None, :]
    s,r,o = test_trip[:,0], test_trip[:,1], test_trip[:,2]
    # get embeddings
    emb_s = model.emb_e(s)
    emb_r = model.emb_rel(r)
    emb_o = model.emb_e(o)
    
    score = model.score_emb(emb_s, emb_r, emb_o)
    emb_s_grad = autograd.grad(score, emb_s)
    epsilon_star = -epsilon * emb_s_grad[0]
    perturbed_emb_s = emb_s + epsilon_star
    
    # get scores for each candidate
    b_begin = 0
    cand_scores = []
    if args.attack_batch_size == -1:
        cand_batch = num_cor
    else:
        cand_batch = args.attack_batch_size
    
    while b_begin < num_cor:
        b_cand_rd = cand_rd[b_begin : b_begin+cand_batch]
        b_cand_od = cand_od[b_begin : b_begin+cand_batch]
        
        rd = torch.from_numpy(b_cand_rd).to(device)
        od = torch.from_numpy(b_cand_od).to(device)
#         sd = torch.from_numpy(np.full(shape=b_cand_rd, fill_value=s.item())).to(device)
        
#         b_nghbr_trip = torch.from_numpy(b_nghbr_trip).to(device)
#         b_nghbr_s, b_nghbr_r, b_nghbr_o = b_nghbr_trip[:,0], b_nghbr_trip[:,1], b_nghbr_trip[:,2]
        #emb_nghbr_s = model.emb_e(b_nghbr_s)
        emb_cand_r = model.emb_rel(rd)
        emb_cand_o = model.emb_e(od)
        
        perturbed_emb_e = perturbed_emb_s.repeat(b_cand_od.shape[0],1)
        emb_e = emb_s.repeat(b_cand_od.shape[0],1)
        #print(perturbed_emb_e.shape, emb_cand_r.shape)
        #print(emb_s.shape, emb_nghbr_r.shape)
        s1 = model.score_emb(perturbed_emb_e, 
                             emb_cand_r, emb_cand_o) #candidate score after perturbed s
        s2 = model.score_emb(emb_e, emb_cand_r, emb_cand_o) #candidate score
        
        score = lambda2*s1 - lambda3*s2
        score = score.detach().cpu().numpy().tolist()
        cand_scores += score
        
        b_begin += cand_batch
        
    cand_scores = np.array(cand_scores)
    cand_scores = torch.from_numpy(cand_scores).to(device)
    # we want to add the candidate with maximum score
    max_values, argsort = torch.sort(cand_scores, -1, descending=True)
    add_idx = argsort[0].item() # index of candidate to add
    max_val = max_values[0].item() # score of the candidate to add
    
    r_dash = cand_rd[add_idx]
    o_dash = cand_od[add_idx]
    add_trip = [s.item(), r_dash, o_dash]
    
    
    return max_val, add_trip
       
    

def get_nghbr_o_addition(test_trip, model, ents, rels, num_cor, epsilon, lambda2, lambda3):
    # get candidates for addition to neighbourhood of o
    # i.e. get candidates for s' and r'
    cand_rd = rng.choice(a=rels, size = num_cor, replace=True) # this is (num_cor,)
    cand_sd = rng.choice(a=ents, size = num_cor, replace=True) # this is (num_cor,)
    
    # --- Get the perturbed object embedding ---
    test_trip = torch.from_numpy(test_trip).to(device)[None, :]
    s,r,o = test_trip[:,0], test_trip[:,1], test_trip[:,2]
    # get embeddings
    emb_s = model.emb_e(s)
    emb_r = model.emb_rel(r)
    emb_o = model.emb_e(o)
    
    score = model.score_emb(emb_s, emb_r, emb_o)
    emb_o_grad = autograd.grad(score, emb_o)
    epsilon_star = -epsilon * emb_o_grad[0]
    perturbed_emb_o = emb_o + epsilon_star
    
    # get scores for each candidate
    b_begin = 0
    cand_scores = []
    if args.attack_batch_size == -1:
        cand_batch = num_cor
    else:
        cand_batch = args.attack_batch_size
    
    while b_begin < num_cor:
        b_cand_rd = cand_rd[b_begin : b_begin+cand_batch]
        b_cand_sd = cand_sd[b_begin : b_begin+cand_batch]
        
        rd = torch.from_numpy(b_cand_rd).to(device)
        sd = torch.from_numpy(b_cand_sd).to(device)
        
#         b_nghbr_trip = nghbr_trip[b_begin : b_begin+nghbr_batch]
#         b_nghbr_trip = torch.from_numpy(b_nghbr_trip).to(device)
#         b_nghbr_s, b_nghbr_r, b_nghbr_o = b_nghbr_trip[:,0], b_nghbr_trip[:,1], b_nghbr_trip[:,2]
        
        
        emb_cand_s = model.emb_e(sd)
        emb_cand_r = model.emb_rel(rd)
        #emb_nghbr_o = model.emb_e(b_nghbr_o)
        
        perturbed_emb_e = perturbed_emb_o.repeat(b_cand_sd.shape[0],1)
        emb_e = emb_o.repeat(b_cand_sd.shape[0],1)
        s1 = model.score_emb(emb_cand_s, 
                             emb_cand_r, perturbed_emb_e) #candidate score after perturbed s
        
        s2 = model.score_emb(emb_cand_s, emb_cand_r, emb_e) #candidate score
        
        score = lambda2*s1 - lambda3*s2
        score = score.detach().cpu().numpy().tolist()
        cand_scores += score
        
        b_begin += cand_batch
        
    cand_scores = np.array(cand_scores)
    cand_scores = torch.from_numpy(cand_scores).to(device)
    # we want to add the candidate with maximum score
    max_values, argsort = torch.sort(cand_scores, -1, descending=True)
    add_idx = argsort[0].item() # index of candidate to add
    max_val = max_values[0].item() # score of the candidate to add
    
    r_dash = cand_rd[add_idx]
    s_dash = cand_sd[add_idx]
    add_trip = [s_dash, r_dash, o.item()]
    
    
    return max_val, add_trip
       
    
def get_additions(test_data, model, ents, rels, args):
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))
    
    triples_to_add = []
    for test_idx, test_trip in enumerate(test_data):
        
        max_val_s, add_trip_s = get_nghbr_s_addition(test_trip, model, ents, rels, args.num_cor, 
                                                     args.epsilon, args.lambda2, args.lambda3)
        max_val_o, add_trip_o = get_nghbr_o_addition(test_trip, model, ents, rels, args.num_cor, 
                                                     args.epsilon, args.lambda2, args.lambda3)

        if max_val_s > max_val_o:
            add_trip = add_trip_s
        else:
            add_trip = add_trip_o
    
        
        triple_to_add = add_trip

        triples_to_add.append(triple_to_add)
        if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
            logger.info('Processed test triple {0}'.format(test_idx))
            logger.info('Time taken: {0}'.format(time.time() - start_time))
    logger.info('Time taken to generate edits: {0}'.format(time.time() - start_time))   

    return triples_to_add    



if __name__ == '__main__':


    parser = utils.get_argument_parser()
    parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
    parser.add_argument('--attack-batch-size', type=int, default=1000, help='Batch size for processing neighbours of target')

    parser.add_argument('--epsilon', type=int, default=1, help='Value of epsilon multiplier in IJCAI add attack')
    parser.add_argument('--lambda2', type=int, default=1, help='Value of lambda2 in IJCAI add attack')
    parser.add_argument('--lambda3', type=int, default=1, help='Value of lambda3 in IJCAI add attack')
    
    parser.add_argument('--corruption-factor', type=float, default=5, help='Random downsampling for scoring in percent')


    # In[5]:


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args.target_split = '0_100_1' # which target split to use 
    # #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
    # args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
    # args.rand_run = 1 #  a number assigned to the random run of the experiment
    args.seed = args.seed + (args.rand_run - 1) # default seed is 17

    # args.model = 'distmult'
    # args.data = 'WN18RR'
    # args.reproduce_results = True

    if args.reproduce_results:
        args = utils.set_hyperparams(args)


    # In[7]:


    # Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)


    args.epochs = -1 #no training here
    model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    log_path = 'logs/attack_logs/ijcai_add_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                               args.target_split, args.budget, args.rand_run)


    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
    logger = logging.getLogger(__name__)
    logger.info('-------------------- Edits with IJCAI baseline ----------------------')
    logger.info('corruption_factor: {0}'.format(args.corruption_factor))
    logger.info('rand_run: {0}'.format(args.rand_run))


    data_path = 'data/target_{0}_{1}_{2}'.format(args.model, args.data, args.target_split)

    n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)

    ##### load data####
    data  = utils.load_data(data_path)
    train_data, valid_data, test_data = data['train'], data['valid'], data['test']

    inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
    to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
    inp_f.close()
    to_skip_eval['lhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['lhs'].items()}
    to_skip_eval['rhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['rhs'].items()}



    model = utils.load_model(model_path, args, n_ent, n_rel, device)

    #neighbours = generate_nghbrs(test_data, train_data) 
    # test set is the target set because we loaded data from target_...

    ents = np.asarray(list(ent_to_id.values()))
    rels = np.asarray(list(rel_to_id.values()))
    args.num_cor = np.math.ceil((n_ent*n_rel)*args.corruption_factor / 100)
    
    triples_to_add = get_additions(test_data, model, ents, rels, args)

    #remove duplicate entries
    df = pd.DataFrame(data=triples_to_add)
    df = df.drop_duplicates()
    # print(df.shape)
    trips_to_add = df.values
    # print(trips_to_delete.shape)
    num_duplicates = len(triples_to_add) - trips_to_add.shape[0]
    # print(num_duplicates)
    
    per_tr = np.concatenate((trips_to_add, train_data))
    
    #remove duplicate entries
    df = pd.DataFrame(data=per_tr)
    df = df.drop_duplicates()
    # print(df.shape)
    per_tr_1 = df.values
    # print(trips_to_delete.shape)
    num_duplicates_1 = per_tr.shape[0] - per_tr_1.shape[0]
    # print(num_duplicates)
    


    logger.info('Shape of perturbed training set: {0}'.format(per_tr_1.shape))
    logger.info('Number of duplicate adversarial additions: {0}'.format(num_duplicates))
    logger.info('Number of adversarial additions already in train data: {0}'.format(num_duplicates_1))


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(per_tr_1.shape[0]))


    save_path = 'data/ijcai_add_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                               args.target_split, args.budget, args.rand_run)



    try :
        os.makedirs(save_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            logger.info(e)
            logger.info('Using the existing folder {0} for processed data'.format(save_path))
        else:
            raise


    new_train = per_tr_1
    num_en_or = np.unique(np.concatenate((train_data[:,0], train_data[:,2]))).shape[0]
    num_en_pos = np.unique(np.concatenate((new_train[:,0], new_train[:,2]))).shape[0]


    with open(os.path.join(save_path, 'train.txt'), 'w') as out:
        for item in new_train:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'train.pickle'), 'wb')
    pickle.dump(new_train.astype('uint64'), out)
    out.close()


    with open(os.path.join(save_path, 'entities_dict.json'), 'w') as f:
        f.write(json.dumps(ent_to_id)  + '\n')

    with open(os.path.join(save_path, 'relations_dict.json'), 'w') as f:
        f.write(json.dumps(rel_to_id)  + '\n')


    with open(os.path.join(save_path, 'valid.txt'), 'w') as out:
        for item in valid_data:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'valid.pickle'), 'wb')
    pickle.dump(valid_data.astype('uint64'), out)
    out.close()


    with open(os.path.join(save_path, 'test.txt'), 'w') as out:
        for item in test_data:
            out.write("%s\n" % "\t".join(map(str, item)))

    out = open(os.path.join(save_path, 'test.pickle'), 'wb')
    pickle.dump(test_data.astype('uint64'), out)
    out.close()


    with open(os.path.join(save_path, 'stats.txt'), 'w') as f:
        f.write('Model: {0} \n'.format(args.model))
        f.write('Data: {0} \n'.format(args.data))
#         f.write('Similarity metric used for addition: {0} \n'.format(args.sim_metric))
        f.write('Length of original training set: {0} \n'. format(train_data.shape[0]))
        f.write('Length of new poisoned training set: {0} \n'. format(new_train.shape[0]))
        f.write('Number of duplicate additions: {0} \n'. format(num_duplicates))
        f.write('Number of additions already in train data: {0} \n'. format(num_duplicates_1))
        f.write('Number of entities in original training set: {0} \n'. format(num_en_or))
        f.write('Number of entities in poisoned training set: {0} \n'. format(num_en_pos))
        f.write('Length of original test set: {0} \n'. format(test_data.shape[0]))
        f.write('---------------------------------------------------------------------- \n')


    with open(os.path.join(save_path, 'influential_triples.txt'), 'w') as out:
        for item in triples_to_add:
            out.write("%s\n" % "\t".join(map(str, item)))


    with open(os.path.join(save_path, 'additions.txt'), 'w') as out:
        for item in trips_to_add:
            out.write("%s\n" % "\t".join(map(str, item)))


            
            
            
    # In[ ]:




