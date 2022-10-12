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


def get_decoy_o_addition(triple, model, train_data, composition_relation):
    '''
    Additions for decoy triples on object side, 
    i.e. [s,r,o'] is the decoy triple
    o' is selected as the entity that is ranked just worse than o for queries (s,r,?)
    '''
    triple = torch.from_numpy(triple).to(device)[None,:]
    s,r,o = triple[:,0], triple[:,1], triple[:,2]
    r1 = torch.from_numpy(np.array([composition_relation[r.item()][0]])).to(device)
    r2 = torch.from_numpy(np.array([composition_relation[r.item()][1]])).to(device)
    
    # filters are (s,r,*)
    # these filters are needed to filter out decoy triples
    filter_o = to_skip_eval['rhs'][(s.item(), r.item())]
    
    pred_o = model.score_sr(s, r, sigmoid=False).squeeze()
    
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
    #argsort_o = argsort_o.cpu().numpy()
    rank_o = torch.where(argsort_o==o.item())[0][0]
    
    if rank_o + 1 >= argsort_o.shape[0]:  # this happens for nations TransE
        r_int = rng.choice(a=np.arange(argsort_o.shape[0]-1), size = 1, replace=True)[0]
        rank_o = r_int - 1
    o_dash = argsort_o[rank_o+1][None]
    
    # filters are (s,r1,?) and (?,r2, o') 
    filter_r = train_data[np.where((train_data[:,0] == s.item()) 
                                       & (train_data[:,1] == r1.item())), 2].squeeze()
    filter_l = train_data[np.where((train_data[:,2] == o_dash.item()) 
                                       & (train_data[:,1] == r2.item())), 0].squeeze()

    #pred_r = model.forward(s,r1, mode='rhs', sigmoid=True) #this gives probabilities
    #pred_l = model.forward(o_dash,r2 ,mode='lhs', sigmoid=True) # this gives probabilities
    pred_r = model.score_sr(s, r1, sigmoid=True)
    pred_l = model.score_or(o_dash, r2, sigmoid=True)
    pred_r, pred_l = pred_r.squeeze(), pred_l.squeeze()
    
    soft_scores = pred_l * pred_r
    soft_scores[filter_l] = -1e6
    soft_scores[filter_r] = -1e6
    #soft_scores[s] = -1e6
    soft_scores[o] = -1e6 # we don't want to select (s,r1,o) as edit

    o_ddash = torch.argmax(soft_scores)
    
    o_dash, o_ddash = o_dash.item(), o_ddash.item()
    _s,_r, _r1, _r2, _o = s.item(), r.item(), r1.item(), r2.item(), o.item()

    decoy_o = [_s, _r, o_dash]

    adv_o_l = [_s, _r1,  o_ddash]
    adv_o_r = [o_ddash, _r2, o_dash]
    
    
    return rank_o.item(), adv_o_l, adv_o_r, decoy_o


def get_decoy_s_addition(triple, model, train_data, composition_relation):
    '''
    Additions for decoy triples on the subject side,
    i.e. [s',r,o] is the decoy triple
    s' is selected as the entity that is ranked just worse than s for queries (?, r, o)
    '''
    triple = torch.from_numpy(triple).to(device)[None,:]
    s,r,o = triple[:,0], triple[:,1], triple[:,2]
    r1 = torch.from_numpy(np.array([composition_relation[r.item()][0]])).to(device)
    r2 = torch.from_numpy(np.array([composition_relation[r.item()][1]])).to(device)
    
    # filters are (*,r, o) 
    # these filters are needed to filter out decoy triples
    filter_s = to_skip_eval['lhs'][(o.item(), r.item())]
    
    pred_s = model.score_or(o, r, sigmoid=False).squeeze()
    
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
    #argsort_s = argsort_s.cpu().numpy()
    rank_s = torch.where(argsort_s==s.item())[0][0]
    
        
    s_dash = argsort_s[rank_s+1][None]
    # filters are (s',r1,?) and (?,r2, o) 
    filter_r = train_data[np.where((train_data[:,0] == s_dash.item()) 
                                       & (train_data[:,1] == r1.item())), 2].squeeze()
    filter_l = train_data[np.where((train_data[:,2] == o.item()) 
                                       & (train_data[:,1] == r2.item())), 0].squeeze()

    #pred_r = model.forward(s_dash,r1, mode='rhs', sigmoid=True) #this gives probabilities
    #pred_l = model.forward(o,r2,mode='lhs', sigmoid=True) # this gives probabilities
    pred_r = model.score_sr(s_dash, r1, sigmoid=True)
    pred_l = model.score_or(o, r2, sigmoid=True)
    pred_r, pred_l = pred_r.squeeze(), pred_l.squeeze()

    soft_scores = pred_l * pred_r
    soft_scores[filter_l] = -1e6
    soft_scores[filter_r] = -1e6
    soft_scores[s] = -1e6 # we don't want to select (s,r2,o) as edit
    #soft_scores[o] = -1e6 

    s_ddash = torch.argmax(soft_scores)
    
    s_dash, s_ddash = s_dash.item(), s_ddash.item()
    _s,_r, _r1, _r2, _o = s.item(), r.item(), r1.item(), r2.item(), o.item()

    decoy_s = [s_dash, _r, _o]

    adv_s_l = [s_dash, _r1,  s_ddash]
    adv_s_r = [s_ddash, _r2,  _o]
    
    
    return rank_s.item(), adv_s_l, adv_s_r, decoy_s
    

def get_additions(train_data, test_data, model):
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))
    
    # #### Pseudocode 
    # 1. composition of a relation is selected by taking the Euclidean distance between r and r1.r2 (for multiplicative models) and r1+r2 (for additive models)
    #     
    # 2. select decoy triple that is ranked just worse than target triple
    # 3. Select entity substitution based on the t-norm of (s,r1,o'') ^ (o'',r2,o')
    # 4. use the maximum t-norm for edit 
    # 5. (s,r1,o'') and (o'',r2,o') are the adversarial edits
    
    r1 = model.emb_rel.weight.data #this is (|R|, k)
    r2 = model.emb_rel.weight.data #this is (|R|, k)
    
    r1,r2 = r1.cpu().numpy(), r2.cpu().numpy()
    
    # all possible compositions of relations
    composed_rel = []
    for relation in r1:
        if args.model == 'transe':
            comp = relation + r2
        else:
            comp = relation * r2
        composed_rel.append(comp)
    composed_rel = np.array(composed_rel) 
    composed_rel = composed_rel.reshape(-1,composed_rel.shape[-1]) # shape is (|R|x|R|, k)
    # the above reshape will have all rows for r10 composed with full r2, then r11 with r2 and so on
    # (composition operations are commutative; so not much worry needed for reshape ordering)
    
    composition_relation = {}
    for rel_id, relation in enumerate(r1):
        dist = relation - composed_rel
        euclidean_dist = np.linalg.norm(dist, axis=1) # shape is (|R|x|R|)
        euclidean_dist = euclidean_dist.reshape(n_rel, -1) # shape is (|R| , |R|)

        candidate_r = np.unravel_index(np.argmin(euclidean_dist), shape=euclidean_dist.shape)
        composition_relation[rel_id] = candidate_r

        if rel_id % 100 ==0:
            logger.info('Processing relation number: {0} '.format(rel_id))
    
    triples_to_add = []
    decoy_triples = []
    summary_dict = {}
    #if args.add_reciprocals:
    #    add_rec = n_rel
    #else:
    #    add_rec = 0
    for test_idx, test_trip in enumerate(test_data):
        
        max_val_s, add_trip_s_l, add_trip_s_r, decoy_s = get_decoy_s_addition(test_trip, model, train_data, composition_relation)
        max_val_o, add_trip_o_l, add_trip_o_r, decoy_o = get_decoy_o_addition(test_trip, model, train_data, composition_relation)

        if max_val_s > max_val_o: ## worse rank implies less confidence
            add_trip_l = add_trip_s_l
            add_trip_r = add_trip_s_r
            decoy_trip = decoy_s
            add_type = 's'
        else:
            add_trip_l = add_trip_o_l
            add_trip_r = add_trip_o_r
            decoy_trip = decoy_o
            add_type = 'o'
    
        
        triples_to_add.append(add_trip_l)
        triples_to_add.append(add_trip_r)
        decoy_triples.append(decoy_trip)
        summary_list = []
        summary_list.append(add_type)
        summary_list.append(list(map(int, add_trip_l)))
        summary_list.append(list(map(int, add_trip_r)))
        summary_dict[test_idx] = summary_list
        
        if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
            logger.info('Processed test triple {0}'.format(test_idx))
            logger.info('Time taken: {0}'.format(time.time() - start_time))
    logger.info('Time taken to generate edits: {0}'.format(time.time() - start_time))   

    return triples_to_add, decoy_triples, summary_dict, composition_relation  


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
    log_path = 'logs/attack_logs/com_add_2_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                               args.target_split, args.budget, args.rand_run)
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
    logger = logging.getLogger(__name__)
    logger.info('-------------------- Edits with Composition Attack - worse ranks ----------------------')
    
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
    
    triples_to_add, decoy_triples, summary_dict, composition_relation = get_additions(train_data, test_data, model)
    
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


    save_path = 'data/com_add_2_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
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
        f.write('----------------------Composition Add 2 (Worse Ranks)------------------------------- \n')


    with open(os.path.join(save_path, 'additions.txt'), 'w') as out: ### note the difference from IJCAI_add - triples_to_add
        for item in triples_to_add:
            out.write("%s\n" % "\t".join(map(str, item)))
            
    with open(os.path.join(save_path, 'decoy_test.txt'), 'w') as out:
        for item in decoy_triples:
            out.write("%s\n" % "\t".join(map(str, item)))
            
    with open(os.path.join(save_path, 'summary_edits.json'), 'w') as out:
        out.write(json.dumps(summary_dict)  + '\n')
        
    composition_relation = {int(k):(int(v[0]), int(v[1])) for k,v in composition_relation.items()}
    
    with open(os.path.join(save_path, 'composed_relations.json'), 'w') as out:
        out.write(json.dumps(composition_relation)  + '\n')
    
    
    
