#!/usr/bin/env python
# coding: utf-8

# This is the notebook to generate adversarial deletions using **Gradient Rollback**. This requires that an Influence Map was saved while training the original model.
# 
# The Influence Map will be loaded for all candidates of the target triple and their influence is given by the difference of predicted scores from learned parameters versus (learned params - influences)
# 
# Duplicate influential triples are allowed

# In[1]:


import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import operator

import json
import torch
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



def generate_nghbrs(test_set, train_set):
    '''
    For every triple in test set, return the index of 
    neighbouring triple in training set,
    i.e. indices in training set are returned
    '''
    n_dict = {}
    for t, triple in enumerate(test_set):
        sub = triple[0]
        obj = triple[2]
        mask = (np.isin(train_set[:,0], [sub, obj]) | np.isin(train_set[:,2], [sub, obj]))
        #nghbrs_dict[t] = pro_train[mask]
        mask_idx = np.where(mask)[0]
        n_dict[t] = mask_idx
    
    return n_dict 

def load_influence_map(args):
    # load the influence map
    influence_path = 'influence_maps/{0}_{1}.pickle'.format(args.data, model_name)
#     with open(influence_path, 'r') as out:
#         influence_map = json.load(out)
    with open(influence_path, "rb") as fp:   # Unpickling
        influence_map = pickle.load(fp)
        
    # From GR codebase
    # https://github.com/carolinlawrence/gradient-rollback/blob/6a26b2ff78394ae91b4cd165806eb95442fa4466/gr/xai/explanation_generator.py#L228
    
    modified_influence_map = defaultdict(dict)
    
    logger.info('------ Modifying Influence Map -----')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))
    for key in list(influence_map):
        # The key is structured as
        # [value of s]_[value of r]_[value of o]_[s or r or o]
        # Eg. '9_24_11_s' is the key for influence of subject in triple (9,24,11)
        elements = key.split('_')
        entity_dim = '%s'%(elements[-1]) # this will be s or r or o
        triple = '%s:%s:%s' % (elements[0], elements[1], elements[2])
        modified_influence_map[triple][entity_dim] = np.array(influence_map[key], dtype=np.float32)
        influence_map.pop(key, None)

    influence_map = modified_influence_map
    logger.info('Time taken to modify IF map: {0}'.format(str(time.time() - start_time)))
    del modified_influence_map
    
    return influence_map


def get_deletions(train_data, test_data, neighbours, model, attack_batch_size, args):
    #load influence map
    influence_map = load_influence_map(args)
    
    # save original model params
    original_emb_e = model.emb_e.weight.data.clone()
    original_emb_rel = model.emb_rel.weight.data.clone()
    
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))
    
    triples_to_delete = []
    if_scores = {}
    for test_idx, test_trip in enumerate(test_data):
        test_trip = test_trip[None, :] # add a batch dimension
        test_trip = torch.from_numpy(test_trip).to(device)
        s_t, r_t, o_t = test_trip[:,0], test_trip[:,1], test_trip[:,2]
        original_score = model.score_triples(s_t,r_t,o_t)
        # GR codebase uses softmax on the array of all possible object scores, but can't use that for single score
        original_prob = torch.sigmoid(original_score)
        
        test_nghbrs = neighbours[test_idx] # these include neighboours of subject and object only
        nghbr_trip = train_data[test_nghbrs]  # test_nghbrs contains indices for training data
        influences = {}
        
        for nghbr_idx in test_nghbrs:
            # get the influence of head, relation and tail
            nghbr = train_data[nghbr_idx] # nghbrs contains indices for training data
            head, rel, tail = nghbr[0], nghbr[1], nghbr[2]
            nghbr = '{0}:{1}:{2}'.format(nghbr[0], nghbr[1], nghbr[2])
            influence_nghbr = influence_map[nghbr]
            influence_head = influence_nghbr['s']
            influence_rel = influence_nghbr['r']
            influence_tail = influence_nghbr['o']
            
            with torch.no_grad():
                modify_emb_e = model.emb_e.weight.data.clone().cpu().detach().numpy()
                modify_emb_rel = model.emb_rel.weight.data.clone().cpu().detach().numpy()
                # we want to modify the embeddings of neighbour
                modify_emb_e[head] = modify_emb_e[head] - influence_head
                modify_emb_e[tail] = modify_emb_e[tail] - influence_tail
                modify_emb_rel[rel] = modify_emb_rel[rel] - influence_rel 
                #modify_emb_rel is actually irrelevant here because neighboours are restricted to entities
                # changing the relation emb matrix will not change the score of target triple

                modify_emb_e = torch.from_numpy(modify_emb_e).to(device)
                modify_emb_rel = torch.from_numpy(modify_emb_rel).to(device)
                model.emb_e.weight.data.copy_(modify_emb_e)
                model.emb_rel.weight.data.copy_(modify_emb_rel)
                score_minus_influence = model.score_triples(s_t,r_t,o_t)
                # score of target with modified params
                prob_minus_influence = torch.sigmoid(score_minus_influence)
                influences[nghbr_idx] = (original_prob - prob_minus_influence).item()

                model.emb_e.weight.data.copy_(original_emb_e)
                model.emb_rel.weight.data.copy_(original_emb_rel)
            
        # get the neighbour with maximum influence
        if_idx = max(influences.items(), key=operator.itemgetter(1))[0]
        triple_to_delete = train_data[if_idx]
        triples_to_delete.append(triple_to_delete)
        
        if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
            logger.info('Processed test triple {0}'.format(str(test_idx)))
            logger.info('Time taken: {0}'.format(str(time.time() - start_time)))
    logger.info('Time taken to generate edits: {0}'.format(str(time.time() - start_time)))    
    
    return triples_to_delete




if __name__ == '__main__':


    parser = utils.get_argument_parser()
    parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
    parser.add_argument('--attack-batch-size', type=int, default=-1, help='Batch size for processing neighbours of target')


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args.target_split = '0_100_1' # which target split to use 
    #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
    # args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
    # args.rand_run = 1 #  a number assigned to the random run of the experiment
    args.seed = args.seed + (args.rand_run - 1) # default seed is 17

    # args.model = 'distmult'
    # args.data = 'WN18RR'
    # args.reproduce_results = True

    if args.reproduce_results:
        args = utils.set_hyperparams(args)



    # Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)


    args.epochs = -1 #no training here
    model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    log_path = 'logs/attack_logs/gr_del_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                               args.target_split, args.budget, args.rand_run)


    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
    logger = logging.getLogger(__name__)


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

    neighbours = generate_nghbrs(test_data, train_data) 
    # test set is the target set because we loaded data from target_...


    # Pseudocode - 
    # - For every target triple, get the relevant neighbours:
    #     - Get the score of target triple from loaded model params
    #     - Save the original param matrix
    #     - For every neighbour:
    #         - Get the influence vector from the influence map
    #         - Update the model params = original params - influence vector
    #         - Get the new score of target triple from updated model params
    #         - Influence for neighbour = difference between scores
    #     - Re-update the param matrix to original value


    triples_to_delete = get_deletions(train_data, test_data, neighbours, 
                                      model, args.attack_batch_size,
                                     args)


    df = pd.DataFrame(data=triples_to_delete)
    df = df.drop_duplicates()
    # print(df.shape)
    trips_to_delete = df.values
    # print(trips_to_delete.shape)
    num_duplicates = len(triples_to_delete) - trips_to_delete.shape[0]
    # print(num_duplicates)

    per_tr_1, n_ignored_edits = utils.perturb_data(train_data, 
                                                   trips_to_delete)


    # Perturbed dataset
    logger.info('Shape of perturbed training set: {0}'.format(per_tr_1.shape))
    logger.info('Number of adversarial deletions ignored (because of singleton nodes): {0}'.format(n_ignored_edits))
    logger.info('Number of duplicate adversarial deletions : {0}'.format(num_duplicates))


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(per_tr_1.shape[0]))


    save_path = 'data/gr_del_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
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
        f.write('Length of original training set: {0} \n'. format(train_data.shape[0]))
        f.write('Length of new poisoned training set: {0} \n'. format(new_train.shape[0]))
        f.write('Number of duplicate deletions: {0} \n'. format(num_duplicates))
        f.write('Number of deletions ignored due to singleton nodes: {0} \n'. format(n_ignored_edits))
        f.write('Number of entities in original training set: {0} \n'. format(num_en_or))
        f.write('Number of entities in poisoned training set: {0} \n'. format(num_en_pos))
        f.write('Length of original test set: {0} \n'. format(test_data.shape[0]))
        f.write('---------------------------------------------------------------------- \n')


    with open(os.path.join(save_path, 'influential_triples.txt'), 'w') as out:
        for item in triples_to_delete:
            out.write("%s\n" % "\t".join(map(str, item)))


    with open(os.path.join(save_path, 'deletions.txt'), 'w') as out:
        for item in trips_to_delete:
            out.write("%s\n" % "\t".join(map(str, item)))







