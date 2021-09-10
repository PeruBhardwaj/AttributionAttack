#!/usr/bin/env python
# coding: utf-8

# In[1]:

### Load the influential triples from deletions and replace the subject or object with most distant entity in L2 metric

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


def get_additions(model, train_data, test_data, if_data):
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))

    ent_emb = model.emb_e.weight
    rel_emb = model.emb_rel.weight

    triples_to_add = []
    for test_idx, test_trip in enumerate(test_data):
        test_trip = torch.from_numpy(test_trip).to(device)[None,:]
        test_s, test_r, test_o = test_trip[:,0], test_trip[:,1], test_trip[:,2]
    #     test_vec = model.score_triples_vec(test_s, test_r, test_o)

        if_trip = if_data[test_idx]
        if_trip = torch.from_numpy(if_trip).to(device)[None,:]
        if_s, if_r, if_o = if_trip[:,0], if_trip[:,1], if_trip[:,2]

        if (if_o == test_s or if_o == test_o):
            # object of IF triple is neighbour - edit will be [s_dash, if_r, if_o]
            if_s_emb = model.emb_e(if_s).squeeze(dim=1)
            #if_r_emb = model.emb_rel(if_r).squeeze(dim=1)
            #cos_sim_s = F.cosine_similarity(if_s_emb, ent_emb)
            cos_sim_s = -torch.norm((ent_emb - if_s_emb), p=2, dim=-1) #L2 similarity between embeddings
            #cos_sim_r = F.cosine_similarity(if_r_emb, rel_emb)

            # filter for (s_dash, r, o), i.e. ignore s_dash that already exist
            filter_s = train_data[np.where((train_data[:,2] == if_o.item()) 
                                                   & (train_data[:,1] == if_r.item())), 0].squeeze()
            #filter_r = train_data[np.where((train_data[:,0] == if_s.item()) 
            #                                      & (train_data[:,2] == if_o.item())), 1].squeeze()
            cos_sim_s[filter_s] = 1e6
            #cos_sim_r[filter_r] = 1e6

            # sort and rank - smallest cosine similarity means largest cosine distance
            # Hence, corrupted entity = one with smallest cos similarity
            min_values_s, argsort_s = torch.sort(cos_sim_s, -1, descending=False)
            #min_values_r, argsort_r = torch.sort(cos_sim_r, -1, descending=False)
            s_dash = argsort_s[0][None, None]
            #r_dash = argsort_r[0][None, None]

            add_trip = [s_dash.item(), if_r.item(), if_o.item()]
            #add_trip = similarity_func[args.sim_metric](model, test_trip, if_trip, s_dash, r_dash, nghbr='o')

        elif (if_s == test_s or if_s == test_o):
            #print('s is neighbour')
            # subject of IF triple is neighbour - edit will be [if_s, if_r, o_dash]
            if_o_emb = model.emb_e(if_o).squeeze(dim=1)
            #if_r_emb = model.emb_rel(if_r).squeeze(dim=1)
            #cos_sim_o = F.cosine_similarity(if_o_emb, ent_emb)
            cos_sim_o = -torch.norm((ent_emb - if_o_emb), p=2, dim=-1) #L2 similarity between embeddings
            #cos_sim_r = F.cosine_similarity(if_r_emb, rel_emb)
            
            # filter for (s, r, o_dash), i.e. ignore o_dash that already exist
            filter_o = train_data[np.where((train_data[:,0] == if_s.item()) 
                                                   & (train_data[:,1] == if_r.item())), 2].squeeze()
            #filter_r = train_data[np.where((train_data[:,0] == if_s.item()) 
            #                                      & (train_data[:,2] == if_o.item())), 1].squeeze()
            cos_sim_o[filter_o] = 1e6
            #cos_sim_r[filter_r] = 1e6

            # sort and rank - smallest cosine similarity means largest cosine distance
            # Hence, corrupted entity = one with smallest cos similarity
            min_values_o, argsort_o = torch.sort(cos_sim_o, -1, descending=False)
            #min_values_r, argsort_r = torch.sort(cos_sim_r, -1, descending=False)
            o_dash = argsort_o[0][None, None]
            #r_dash = argsort_r[0][None, None]

            add_trip = [if_s.item(), if_r.item(), o_dash.item()]
            #add_trip = similarity_func[sim_metric](model, test_trip, if_trip, o_dash, r_dash, nghbr='s')

        else:
            logger.info('Unexpected behaviour')

        triples_to_add.append(add_trip)

    if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
        logger.info('Processed test triple {0}'.format(str(test_idx)))
        logger.info('Time taken: {0}'.format(str(time.time() - start_time)))
    logger.info('Time taken to generate edits: {0}'.format(str(time.time() - start_time)))
    
    return triples_to_add


# In[2]:


parser = utils.get_argument_parser()
parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
parser.add_argument('--attack-batch-size', type=int, default=-1, help='Batch size for processing neighbours of target')

#Budget, target-split and rand-run values are to select the attack dataset to load influential triples
parser.add_argument('--sim-metric', type=str, default='cos', help='Value of similarity metric to use')


# In[3]:


# import sys
# sys.argv = ['prog.py']


# In[4]:


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args.target_split = '0_100_1' # which target split to use 
# #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
# args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
# args.rand_run = 1 #  a number assigned to the random run of the experiment
args.seed = args.seed + (args.rand_run - 1) # default seed is 17

# args.model = 'distmult'
# args.data = 'FB15k-237'
# args.sim_metric = 'cos'
# args.reproduce_results = True

if args.reproduce_results:
    args = utils.set_hyperparams(args)


# In[5]:


# Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)
rng = np.random.default_rng(seed=args.seed)


# In[6]:


args.epochs = -1 #no training here
model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
log_path = 'logs/attack_logs/{5}_add_5_l2_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                           args.target_split, args.budget, args.rand_run,
                                                                 args.sim_metric
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
if_data_path = 'data/{5}_del_{0}_{1}_{2}_{3}_{4}'.format(args.model, args.data, 
                                                        args.target_split, 
                                                        args.budget, args.rand_run,
                                                        args.sim_metric)


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


###### load influential triples #######
df = pd.read_csv(os.path.join(if_data_path, 'influential_triples.txt'), sep='\t', 
                 header=None, names=None, dtype=int)
#df = df.drop_duplicates() -- don't want to drop influential triples
if_triples = df.values


# In[10]:


model = utils.load_model(model_path, args, n_ent, n_rel, device)


triples_to_add = get_additions(model, train_data, test_data, if_triples)


# In[14]:


#remove duplicate entries
df = pd.DataFrame(data=triples_to_add)
df = df.drop_duplicates()
# print(df.shape)
trips_to_add = df.values
# print(trips_to_delete.shape)
num_duplicates = len(triples_to_add) - trips_to_add.shape[0]
# print(num_duplicates)


# In[15]:


per_tr = np.concatenate((trips_to_add, train_data))


# In[16]:


#remove duplicate entries
df = pd.DataFrame(data=per_tr)
df = df.drop_duplicates()
# print(df.shape)
per_tr_1 = df.values
# print(trips_to_delete.shape)
num_duplicates_1 = per_tr.shape[0] - per_tr_1.shape[0]
# print(num_duplicates)


# In[17]:


logger.info('Shape of perturbed training set: {0}'.format(per_tr_1.shape))
logger.info('Number of duplicate adversarial additions: {0}'.format(num_duplicates))
logger.info('Number of adversarial additions already in train data: {0}'.format(num_duplicates_1))


logger.info ('Length of original training set: ' + str(train_data.shape[0]))
logger.info ('Length of new poisoned training set: ' + str(per_tr_1.shape[0]))


# In[18]:


save_path = 'data/{5}_add_5_l2_{0}_{1}_{2}_{3}_{4}'.format(args.model, args.data, 
                                                    args.target_split, args.budget, args.rand_run,
                                                    args.sim_metric
                                                                )


# In[19]:


try :
    os.makedirs(save_path)
except OSError as e:
    if e.errno == errno.EEXIST:
        logger.info(e)
        logger.info('Using the existing folder {0} for processed data'.format(save_path))
    else:
        raise


# In[20]:


new_train = per_tr_1
num_en_or = np.unique(np.concatenate((train_data[:,0], train_data[:,2]))).shape[0]
num_en_pos = np.unique(np.concatenate((new_train[:,0], new_train[:,2]))).shape[0]


# In[21]:


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


# In[22]:


with open(os.path.join(save_path, 'stats.txt'), 'w') as f:
        f.write('Model: {0} \n'.format(args.model))
        f.write('Data: {0} \n'.format(args.data))
        f.write('Similarity metric used for influential triples: {0} \n'.format(args.sim_metric))
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
















