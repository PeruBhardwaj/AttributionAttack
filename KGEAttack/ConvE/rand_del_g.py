#!/usr/bin/env python
# coding: utf-8

# This is the notebook for randomly deleting triples from the graph. 
# 
# The number of triples deleted is the number of target triples, but deletion for each target triple is not restricted to its neighbourhood

# In[ ]:



# In[3]:


import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import operator
import sys

import time
import json
import logging
import argparse 
import math
from pprint import pprint
import errno

from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe
import utils


def get_deletions(test_data, train_data, rng):
    logger.info('----- Generating deletions ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))
    triples_to_delete = []
    for test_idx, test_trip in enumerate(test_data):
        rnd_idx = rng.choice(a=train_data.shape[0], size = 1, replace=False)[0]
        #select a random index in neighbourhood
        rnd_trip = train_data[rnd_idx]
        triples_to_delete.append(rnd_trip)
        if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
            logger.info('Processed test triple {0}'.format(str(test_idx)))
            logger.info('Time taken: {0}'.format(str(time.time() - start_time)))
    logger.info('Time taken to generate edits: {0}'.format(str(time.time() - start_time)))
        
    return np.array(triples_to_delete)


if __name__ == '__main__':


    parser = utils.get_argument_parser()
    parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')

    args = parser.parse_args()
    #args.target_split = '0_100_1' # which target split to use 
    #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
    #args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
    #args.rand_run = 1 #  a number assigned to the random run of the experiment
    args.seed = args.seed + (args.rand_run - 1) # default seed is 17

    #args.data = 'WN18RR'
    #args.model = 'complex'

    # Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
    #torch.manual_seed(args.seed)
    #cudnn.deterministic = True
    #cudnn.benchmark = False
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)


    log_path = 'logs/attack_logs/rand_del_g_{0}_{1}_{2}_{3}_{4}.log'.format( args.model, args.data, args.target_split, args.budget, args.rand_run)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S',
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


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of target test set: ' + str(test_data.shape[0]))

    triples_to_delete = get_deletions(test_data, train_data, rng)

    df = pd.DataFrame(data=triples_to_delete)
    df = df.drop_duplicates()
    #print(df.shape)
    trips_to_delete = df.values
    #print(trips_to_delete.shape)
    num_duplicates = triples_to_delete.shape[0] - trips_to_delete.shape[0]
    #print(num_duplicates)


    per_tr_1, n_ignored_edits = utils.perturb_data(train_data, trips_to_delete)

    # Perturbed dataset
    logger.info('Shape of perturbed training set: {0}'.format(per_tr_1.shape))
    logger.info('Number of adversarial deletions ignored (because of singleton nodes): {0}'.format(n_ignored_edits))
    logger.info('Number of duplicate adversarial deletions : {0}'.format(num_duplicates))


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(per_tr_1.shape[0]))


    save_path = 'data/rand_del_g_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, args.target_split,args.budget, args.rand_run)


    try :
        os.makedirs(save_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            logger.info(e)
            logger.info('Using the existing folder {0} for processed data'.format(save_path))
        else:
            raise


# In[ ]:


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



    with open(os.path.join(save_path, 'influential_triples.txt'), 'w') as out:
        for item in triples_to_delete:
            out.write("%s\n" % "\t".join(map(str, item)))


    with open(os.path.join(save_path, 'deletions.txt'), 'w') as out:
        for item in trips_to_delete:
            out.write("%s\n" % "\t".join(map(str, item)))

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
    



