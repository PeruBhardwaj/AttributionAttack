#!/usr/bin/env python
# coding: utf-8

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

from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe
import utils


def get_additions(test_data, ents, rels):
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))

    triples_to_add = []
    for test_idx, test_trip in enumerate(test_data):
        test_s, test_r, test_o = test_trip[0], test_trip[1], test_trip[2]
        
        rel_choices = rels[np.where(rels!=test_r)]
        # r needs to be excluded because the target triple should not be added to training data
#         ent_choices_o = ents[np.where(ents!=test_o)]
#         ent_choices_s = ents[np.where(ents!=test_s)]
        
        rand_r = rng.choice(a=rel_choices, size = 1, replace=True)[0]
        rand_o = rng.choice(a=ents, size = 1, replace=True)[0]
        rand_s = rng.choice(a=ents, size = 1, replace=True)[0]
        
        add_trip = [rand_s, rand_r, rand_o]
        
        triples_to_add.append(add_trip)

    if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
        logger.info('Processed test triple {0}'.format(test_idx))
        logger.info('Time taken: {0}'.format(time.time() - start_time))
    logger.info('Time taken to generate edits: {0}'.format(time.time() - start_time))
    
    return triples_to_add



# In[2]:


parser = utils.get_argument_parser()
parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
parser.add_argument('--attack-batch-size', type=int, default=-1, help='Batch size for processing neighbours of target')

#Budget, target-split and rand-run values are to select the attack dataset to load influential triples
# parser.add_argument('--sim-metric', type=str, default='cos', help='Value of similarity metric to use')


# In[3]:


# import sys
# sys.argv = ['prog.py']


# In[4]:


args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args.target_split = '0_100_1' # which target split to use 
# #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
# args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
# args.rand_run = 1 #  a number assigned to the random run of the experiment
args.seed = args.seed + (args.rand_run - 1) # default seed is 17

# args.model = 'distmult'
# args.data = 'WN18RR'
# args.sim_metric = 'l2'
# args.reproduce_results = True

if args.reproduce_results:
    args = utils.set_hyperparams(args)


# In[5]:


# Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
# torch.manual_seed(args.seed)
# cudnn.deterministic = True
# cudnn.benchmark = False
np.random.seed(args.seed)
rng = np.random.default_rng(seed=args.seed)


# In[6]:


args.epochs = -1 #no training here
model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
log_path = 'logs/attack_logs/rand_add_g_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                           args.target_split, args.budget, args.rand_run
                                                                )


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                    filename = log_path
                       )
logger = logging.getLogger(__name__)


# In[7]:


# load the data from target data
data_path = 'data/target_{0}_{1}_{2}'.format(args.model, args.data, args.target_split)

# load the influential triples from attack data
# if_data_path = 'data/{5}_del_{0}_{1}_{2}_{3}_{4}'.format(args.model, args.data, 
#                                                         args.target_split, 
#                                                         args.budget, args.rand_run,
#                                                         args.sim_metric)


# In[8]:


n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)

##### load data####
data  = utils.load_data(data_path)
train_data, valid_data, test_data = data['train'], data['valid'], data['test']

inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
inp_f.close()
to_skip_eval['lhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['lhs'].items()}
to_skip_eval['rhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['rhs'].items()}


# In[9]:





# In[10]:


ents = np.asarray(list(ent_to_id.values()))
rels = np.asarray(list(rel_to_id.values()))


# In[11]:


triples_to_add = get_additions(test_data, ents, rels)


# In[12]:


#remove duplicate entries
df = pd.DataFrame(data=triples_to_add)
df = df.drop_duplicates()
# print(df.shape)
trips_to_add = df.values
# print(trips_to_delete.shape)
num_duplicates = len(triples_to_add) - trips_to_add.shape[0]
# print(num_duplicates)


# In[13]:


per_tr = np.concatenate((trips_to_add, train_data))


# In[14]:


#remove duplicate entries
df = pd.DataFrame(data=per_tr)
df = df.drop_duplicates()
# print(df.shape)
per_tr_1 = df.values
# print(trips_to_delete.shape)
num_duplicates_1 = per_tr.shape[0] - per_tr_1.shape[0]
# print(num_duplicates)


# In[15]:


logger.info('Shape of perturbed training set: {0}'.format(per_tr_1.shape))
logger.info('Number of duplicate adversarial additions: {0}'.format(num_duplicates))
logger.info('Number of adversarial additions already in train data: {0}'.format(num_duplicates_1))


logger.info ('Length of original training set: ' + str(train_data.shape[0]))
logger.info ('Length of new poisoned training set: ' + str(per_tr_1.shape[0]))


# In[16]:


save_path = 'data/rand_add_g_{0}_{1}_{2}_{3}_{4}'.format(args.model, args.data, 
                                                    args.target_split, args.budget, args.rand_run
                                                                )


# In[17]:


try :
    os.makedirs(save_path)
except OSError as e:
    if e.errno == errno.EEXIST:
        logger.info(e)
        logger.info('Using the existing folder {0} for processed data'.format(save_path))
    else:
        raise


# In[18]:


new_train = per_tr_1
num_en_or = np.unique(np.concatenate((train_data[:,0], train_data[:,2]))).shape[0]
num_en_pos = np.unique(np.concatenate((new_train[:,0], new_train[:,2]))).shape[0]


# In[19]:


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


# In[20]:


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




