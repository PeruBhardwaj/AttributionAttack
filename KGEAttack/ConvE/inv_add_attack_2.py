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
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch.autograd as autograd

from model import Distmult, Complex, Conve, Transe
import utils


def get_decoy_o_addition(triple, model, train_data, inverse_relation):
    '''
    Additions for decoy triples on object side, 
    i.e. [s,r,o'] is the decoy triple
    o' is selected as the entity that is ranked just worse than o for queries (s,r,?)
    '''
    triple = torch.from_numpy(triple).to(device)[None,:]
    s,r,o = triple[:,0], triple[:,1], triple[:,2]
    r_inv = torch.from_numpy(np.array([inverse_relation[r.item()]])).to(device)
    
    # filters are (*,ri, s) - to avoid selecting adversarial triples already in training data
    filter_o = train_data[np.where((train_data[:,2] == s.item()) 
                                           & (train_data[:,1] == r_inv.item())), 0].squeeze()
    filter_o = torch.from_numpy(filter_o)
    
    emb_s = model.emb_e(s).squeeze(dim=1)
    emb_r = model.emb_rel(r).squeeze(dim=1)
    pred_o = model.forward(emb_s, emb_r, mode='rhs', sigmoid=False).squeeze()
    
    # save the prediction that is relevant
    target_value_o = pred_o[o.item()].item()
    # filter the unwanted entities
    # since we want to find rank just worse than target entity, 
    # we set scores for all unwanted entities to large values
    pred_o[filter_o] = 1e6
    
    # write back the saved values in case filter zeroed out these 2 values as well
    pred_o[o.item()] = target_value_o
    
    # sort and rank
    max_values_o, argsort_o = torch.sort(pred_o, -1, descending=True)
    argsort_o = argsort_o.cpu().numpy()
    rank_o = np.where(argsort_o==o.item())[0][0]
    
    s,r,o = s.item(), r.item(), o.item()
    r_inv = r_inv.item()
    o_dash = argsort_o[rank_o+1].item()
    decoy_o = [s,r,o_dash]
    adv_o = [o_dash, r_inv, s]
    
    return rank_o, adv_o, decoy_o


def get_decoy_s_addition(triple, model, train_data, inverse_relation, add_rec):
    '''
    Additions for decoy triples on the subject side,
    i.e. [s',r,o] is the decoy triple
    s' is selected as the entity that is ranked just worse than s for queries (?, r, o)
    '''
    triple = torch.from_numpy(triple).to(device)[None,:]
    s,r,o = triple[:,0], triple[:,1], triple[:,2]
    r_inv = torch.from_numpy(np.array([inverse_relation[r.item()]])).to(device)
    
    # filters are (o,ri,*) - to avoid selecting adversarial triples already in training data
    filter_s = train_data[np.where((train_data[:,0] == o.item()) 
                                           & (train_data[:,1] == r_inv.item())), 2].squeeze()
    filter_s = torch.from_numpy(filter_s)
    
    emb_o = model.emb_e(o).squeeze(dim=1)
    r_rev = r + add_rec
    emb_rrev = model.emb_rel(r_rev).squeeze(dim=1)
    pred_s = model.forward(emb_o, emb_rrev, mode='lhs', sigmoid=False).squeeze()
    
    # save the prediction that is relevant
    target_value_s = pred_s[s.item()].item()
    # filter the unwanted entities
    # since we want to find rank just worse than target entity, 
    # we set scores for all unwanted entities to large values
    pred_s[filter_s] = 1e6
    
    # write back the saved values in case filter zeroed out these 2 values as well
    pred_s[s.item()] = target_value_s

    # sort and rank
    max_values_s, argsort_s = torch.sort(pred_s, -1, descending=True)
    argsort_s = argsort_s.cpu().numpy()
    rank_s = np.where(argsort_s==s.item())[0][0]
    
    s,r,o = s.item(), r.item(), o.item()
    r_inv = r_inv.item()
    s_dash = argsort_s[rank_s+1].item()
    decoy_s = [s_dash, r, o]
    adv_s = [o, r_inv, s_dash]
    
    return rank_s, adv_s, decoy_s
    

def get_additions(train_data, test_data, model, n_rel, args):
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))
    
    # #### Pseudocode 
    # 1. For every relation r1, its inverse relation r2 will be the one that minimizes r1.r2 = 1
    #     
    # 2. For every test triple, choose o' with just worse rank on o-side and choose s' with just worse rank on s-side
    
    r1 = model.emb_rel.weight.data #this is (|R|, k)
    r2 = model.emb_rel.weight.data #this is (|R|, k)
    
    inverse_relation = {}
    if args.model == 'transe':
        # we want r1+r2 ~ 0 for additive models
        inverse_scores = torch.sum(r1[:,None,:] + r2[None, :, :], dim=-1)
    else:
        # we want r1.r2 ~1 for multiplicative models
        inverse_scores = torch.mm(r1, r2.transpose(1,0))
        inverse_scores = 1 - inverse_scores


    inverse_scores = torch.abs(inverse_scores)
    inv_rels = torch.min(inverse_scores, dim=1).indices.cpu().numpy()
    idx = np.arange(r1.shape[0])
    inverse_relation = dict(zip(idx, inv_rels))
    
    triples_to_add = []
    decoy_triples = []
    summary_dict = {}
    if args.add_reciprocals:
        add_rec = n_rel
    else:
        add_rec = 0
    for test_idx, test_trip in enumerate(test_data):
        
        max_val_s, add_trip_s, decoy_s = get_decoy_s_addition(test_trip, model, train_data, inverse_relation, add_rec)
        max_val_o, add_trip_o, decoy_o = get_decoy_o_addition(test_trip, model, train_data, inverse_relation)

        if max_val_s > max_val_o: ## worse rank implies less confidence
            add_trip = add_trip_s
            decoy_trip = decoy_s
            add_type = 's'
        else:
            add_trip = add_trip_o
            decoy_trip = decoy_o
            add_type = 'o'
    
        
        triples_to_add.append(add_trip)
        decoy_triples.append(decoy_trip)
        summary_list = []
        summary_list.append(add_type)
        summary_list.append(list(map(int, add_trip)))
        summary_dict[test_idx] = summary_list
        
        if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
            logger.info('Processed test triple {0}'.format(test_idx))
            logger.info('Time taken: {0}'.format(time.time() - start_time))
    logger.info('Time taken to generate edits: {0}'.format(time.time() - start_time))   

    return triples_to_add, decoy_triples, summary_dict, inverse_relation  


if __name__ == '__main__':


    parser = utils.get_argument_parser()
    parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
    parser.add_argument('--attack-batch-size', type=int, default=1000, help='Batch size for processing neighbours of target')


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
        
    # Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)


    args.epochs = -1 #no training here
    model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    log_path = 'logs/attack_logs/inv_add_2_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                               args.target_split, args.budget, args.rand_run)
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
    logger = logging.getLogger(__name__)
    logger.info('-------------------- Edits with Inverse Attack - worse ranks ----------------------')
    
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
    
    triples_to_add, decoy_triples, summary_dict, inverse_relation = get_additions(train_data, test_data, model, n_rel, args)
    
    # remove duplicate entries in adversarial additions
    df = pd.DataFrame(data=triples_to_add)
    df = df.drop_duplicates()
    trips_to_add = df.values
    num_duplicates = len(triples_to_add) - trips_to_add.shape[0]
    
    per_tr = np.concatenate((trips_to_add, train_data))
    
    # remove duplicate entries in perturbed data
    df = pd.DataFrame(data=per_tr)
    df = df.drop_duplicates()
    per_tr_1 = df.values
    num_duplicates_1 = per_tr.shape[0] - per_tr_1.shape[0]
    
    logger.info('Shape of perturbed training set: {0}'.format(per_tr_1.shape))
    logger.info('Number of duplicate adversarial additions: {0}'.format(num_duplicates))
    logger.info('Number of adversarial additions already in train data: {0}'.format(num_duplicates_1))


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(per_tr_1.shape[0]))


    save_path = 'data/inv_add_2_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
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
        f.write('Number of duplicate additions: {0} \n'. format(num_duplicates))
        f.write('Number of additions already in train data: {0} \n'. format(num_duplicates_1))
        f.write('Number of entities in original training set: {0} \n'. format(num_en_or))
        f.write('Number of entities in poisoned training set: {0} \n'. format(num_en_pos))
        f.write('Length of original test set: {0} \n'. format(test_data.shape[0]))
        f.write('----------------------Inverse Add 2 (Worse Ranks)------------------------------- \n')


    with open(os.path.join(save_path, 'additions.txt'), 'w') as out: ### note the difference from IJCAI_add - triples_to_add
        for item in triples_to_add:
            out.write("%s\n" % "\t".join(map(str, item)))
            
    with open(os.path.join(save_path, 'decoy_test.txt'), 'w') as out:
        for item in decoy_triples:
            out.write("%s\n" % "\t".join(map(str, item)))
            
    with open(os.path.join(save_path, 'summary_edits.json'), 'w') as out:
        out.write(json.dumps(summary_dict)  + '\n')
        
    inverse_relation = {int(k):int(v) for k,v in inverse_relation.items()}
    
    with open(os.path.join(save_path, 'inverse_relations.json'), 'w') as out:
        out.write(json.dumps(inverse_relation)  + '\n')
    
    
    
