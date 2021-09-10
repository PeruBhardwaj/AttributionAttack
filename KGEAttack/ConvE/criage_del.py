#!/usr/bin/env python
# coding: utf-8

# In this notebook, I delete a triple from the neighbourhood of the target triple using the method from criage.
# 
# Neighbourhood refers to the triples that share the entities with target's entities.
# 
# I use criage to separately determine the most influential triple for the subject, then the most influential triple for the object. Then, I choose the more influential of these two triples as the final
# 
# 

# In[1]:


# Duplicate influential triples are allowed


# In[6]:


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


# In[4]:

def sig (x, y):
    return 1 / (1 + np.exp(-np.dot(x, np.transpose(y))))

def point_hess(e_o, nei, embd_e, embd_rel):
    H = np.zeros((e_o.shape[1], e_o.shape[1]))
    for i in nei:
        X = np.multiply(np.reshape(embd_e[i[0]], (1, -1)), np.reshape(embd_rel[i[1]], (1, -1)))
        sig_tri = sig(e_o, X)
        Sig = (sig_tri)*(1-sig_tri)
        H += Sig * np.dot(np.transpose(X), X)
    return H


def point_score(Y, X, e_o, H):
    sig_tri = sig(e_o, X) 
    M = np.linalg.inv(H + (sig_tri)*(1-sig_tri)*np.dot(np.transpose(X), X))
    Score = - np.dot(Y, np.transpose((1-sig_tri)*np.dot(X, M)))
    return Score, M


def find_best_at(pred, E2):
    e2 = E2.view(-1).data.cpu().numpy()
    Pred = pred.view(-1).data.cpu().numpy()
    A1 = np.dot(Pred, e2)
    A2 = np.dot(e2, e2)
    A3 = np.dot(Pred, Pred)
    A = math.sqrt(np.true_divide(A3*A2-0.5, A3*A2-A1**2))
    B = np.true_divide(math.sqrt(0.5)-A*A1, A2) 
    return float(A), float(B)


def find_best_attack(e_o, e_s, Y1, Y2, nei1, nei2, embd_e, embd_rel, model):
    '''
    Return the most influential triple in neighbourhood of s or o;
       depending on whether nei1 has triples or nei2 does
    '''
    dict_s = defaultdict(float)
    if len(nei1) > 0:
        # find influential triple in the neighbourhood of o
        H1 = point_hess(e_o, nei1, embd_e, embd_rel)
        #if len(nei1)> 50:
        #    nei1 = nei1[:50]
        for idx, i in enumerate(nei1):
            e1_or = i[0]
            rel = i[1]
            e1 = torch.cuda.LongTensor([e1_or])
            rel = torch.cuda.LongTensor([rel])
            pred = criage_encoder(model, e1, rel,model_name = args.model).data.cpu().numpy()
            score_t, M = point_score(Y1, pred, e_o , H1)
            dict_s[idx] = score_t[0][0]
    if len(nei2) > 0: 
        # find influential triple in the neighbouhood of s
        H2 = point_hess(e_s, nei2, embd_e, embd_rel) 
        #if len(nei2)> 50:
        #    nei2 = nei2[:50]  
        for idx, i in enumerate(nei2):
            e1_or = i[0]
            rel = i[1]
            e1 = torch.cuda.LongTensor([e1_or])
            rel = torch.cuda.LongTensor([rel])
            pred = criage_encoder(model, e1, rel,model_name = args.model).data.cpu().numpy()
            score_t, M = point_score(Y2, pred, e_s , H2)
            dict_s[idx] = score_t[0][0]

    # I am not sure why sorting is ascending, but its this way in the code on GitHub
    #sorted_score = sorted(dict_s.items(), key=operator.itemgetter(1), reverse=True)
    sorted_score = sorted(dict_s.items(), key=operator.itemgetter(1))

    triple_idx = sorted_score[0][0]
    triple_score = sorted_score[0][1]
    
    if len(nei1)> 0:
        triple = nei1[triple_idx]
    if len(nei2) > 0:
        triple = nei2[triple_idx]

    return triple,triple_score



def criage_encoder(model, sub, rel, model_name='distmult'):
    # https://github.com/pouyapez/criage/blob/bb2d3f44049f70c644442d340c04fbf612758e9a/CRIAGE/model_auto.py#L157
    if model_name == 'conve':
        sub_emb = model.emb_e(sub)
        rel_emb = model.emb_rel(rel)
        stacked_inputs = model.concat(sub_emb, rel_emb)
        stacked_inputs = model.bn0(stacked_inputs)
        #x  = model.inp_drop(stacked_inputs)
        x  = model.conv1(stacked_inputs)
        x  = model.bn1(x)
        x  = F.relu(x)
        #x  = model.feature_drop(x)
        #x  = x.view(x.shape[0], -1)
        x  = x.view(-1, model.flat_sz)
        x  = model.fc(x)
        #x  = self.hidden_drop(x)
        x  = model.bn2(x)
        x  = F.relu(x)
        
        #x  = torch.mm(x, self.emb_e.weight.transpose(1,0))
        #x += self.b.expand_as(x)
        
        #if sigmoid:
            #pred = torch.sigmoid(x)
        #else:
            #pred = x
        return x
    else:
        sub_emb = model.emb_e(sub)
        rel_emb = model.emb_rel(rel)
        sub_emb = sub_emb.squeeze(dim=1)
        rel_emb = rel_emb.squeeze(dim=1)

        #sub_emb = self.inp_drop(sub_emb)
        #rel_emb = self.inp_drop(rel_emb)

        pred = sub_emb*rel_emb
        return pred
    

def get_deletions(train_data, test_data, model):
    emb_e = model.emb_e.weight.data.cpu().numpy()
    emb_rel = model.emb_rel.weight.data.cpu().numpy()
    
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))
    
    triples_to_delete = []
    
    for test_idx, test_trip in enumerate(test_data):

        test_trip = test_trip[None, :] # add a batch dimension
        test_trip = torch.from_numpy(test_trip).to(device)
        e1,rel,e2 = test_trip[:,0], test_trip[:,1], test_trip[:,2]

        pred1 = criage_encoder(model, e1, rel, model_name = args.model)
        pred2 = criage_encoder(model, e2, rel, model_name = args.model)

        E2 = model.emb_e(e2) # object embedding
        E1 = model.emb_e(e1) # subject embedding

        # get the neighbours for object
        nghbr_mask_o = (np.isin(train_data[:,0], [e2.item()]) | np.isin(train_data[:,2], [e2.item()]))
        test_neighbours = np.where(nghbr_mask_o)[0] # this is index of neighbours in training data

        nghbr_trip_o = train_data[test_neighbours] # actual neighbour triples
        nghbr_list_o = nghbr_trip_o.tolist()
        nghbr_list_s = []
        best_o, score_o = find_best_attack(E2.data.cpu().numpy(), E1.data.cpu().numpy(),
                                   pred1.data.cpu().numpy(), pred2.data.cpu().numpy(), 
                                   nghbr_list_o, nghbr_list_s,
                                   emb_e, emb_rel, model)

        # get the neighbours for subject
        nghbr_mask_s = (np.isin(train_data[:,0], [e1.item()]) | np.isin(train_data[:,2], [e1.item()]))
        test_neighbours = np.where(nghbr_mask_s)[0] # this is index of neighbours in training data

        nghbr_trip_s = train_data[test_neighbours] # actual neighbour triples
        nghbr_list_s = nghbr_trip_s.tolist()
        nghbr_list_o = []
        best_s, score_s = find_best_attack(E2.data.cpu().numpy(), E1.data.cpu().numpy(),
                                   pred1.data.cpu().numpy(), pred2.data.cpu().numpy(), 
                                   nghbr_list_o, nghbr_list_s,
                                   emb_e, emb_rel, model)

        if score_o < score_s:
            triple_to_delete = best_o # because ascending sorting is used in find_best_attack
        else:
            triple_to_delete = best_s

        triples_to_delete.append(triple_to_delete)
        
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


# In[8]:

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[10]:
    
    #args.target_split = '0_100_1' # which target split to use 
    #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
    #args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
    #args.rand_run = 1 #  a number assigned to the random run of the experiment
    args.seed = args.seed + (args.rand_run - 1) # default seed is 17

    #args.model = 'distmult'
    #args.data = 'FB15k-237'
    
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
    #log_path = 'logs/inv_add_1_{0}_{1}_{2}_{3}.log'.format(args.data, model_name, args.num_batches, args.epochs)
    log_path = 'logs/attack_logs/criage_del_{0}_{1}_{2}_{3}_{4}.log'.format( args.model, args.data, 
                                                               args.target_split, args.budget, args.rand_run)


    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                        filename=log_path
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


    triples_to_delete = get_deletions(train_data, test_data, model)


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


    save_path = 'data/criage_del_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, args.target_split, args.budget, args.rand_run)



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
        #f.write('Using descending sorting instead of ascending sorting here \n')
        f.write('---------------------------------------------------------------------- \n')



    with open(os.path.join(save_path, 'influential_triples.txt'), 'w') as out:
        for item in triples_to_delete:
            out.write("%s\n" % "\t".join(map(str, item)))


    with open(os.path.join(save_path, 'deletions.txt'), 'w') as out:
        for item in trips_to_delete:
            out.write("%s\n" % "\t".join(map(str, item)))


