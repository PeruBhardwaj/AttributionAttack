'''
In this notebook, I delete a triple from the neighbourhood of the target triple using the Influence Function
The estimator for IHVP uses 1-N training with cross entropy loss (same as original model)
'''

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

def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def get_model_loss(batch, model, device):
    #batch = batch[0].to(device)
    s,r,o = batch[:,0], batch[:,1], batch[:,2]

    emb_s = model.emb_e(s).squeeze(dim=1)
    emb_r = model.emb_rel(r).squeeze(dim=1)
    emb_o = model.emb_e(o).squeeze(dim=1)

    if args.add_reciprocals:
        r_rev = r + n_rel
        emb_rrev = model.emb_rel(r_rev).squeeze(dim=1)
    else:
        r_rev = r
        emb_rrev = emb_r

    pred_sr = model.forward(emb_s, emb_r, mode='rhs')
    loss_sr = model.loss(pred_sr, o) # loss is cross entropy loss

    pred_or = model.forward(emb_o, emb_rrev, mode='lhs')
    loss_or = model.loss(pred_or, s)

    train_loss = loss_sr + loss_or
    return train_loss

def lp_regularizer(model, weight, p):
    # Apply p-norm regularization; assign weights to each param
    trainable_params = [model.emb_e.weight, model.emb_rel.weight]
    norm = 0
    for i in range(len(trainable_params)):
        #norm += weight * trainable_params[i].norm(p = p)**p
        norm += weight * torch.sum( torch.abs(trainable_params[i]) ** p)

    return norm

def n3_regularizer(factors, weight, p):
    # factors are the embeddings for lhs, rel, rhs for triples in a batch
    norm = 0
    for f in factors:
        norm += weight * torch.sum(torch.abs(f) ** p)

    return norm / factors[0].shape[0] # scale by number of triples in batch

def get_train_loss(batch, model, device, args):
    #batch = batch[0].to(device)
    s,r,o = batch[:,0], batch[:,1], batch[:,2]

    emb_s = model.emb_e(s).squeeze(dim=1)
    emb_r = model.emb_rel(r).squeeze(dim=1)
    emb_o = model.emb_e(o).squeeze(dim=1)

    if args.add_reciprocals:
        r_rev = r + n_rel
        emb_rrev = model.emb_rel(r_rev).squeeze(dim=1)
    else:
        r_rev = r
        emb_rrev = emb_r

    pred_sr = model.forward(emb_s, emb_r, mode='rhs')
    loss_sr = model.loss(pred_sr, o) # loss is cross entropy loss

    pred_or = model.forward(emb_o, emb_rrev, mode='lhs')
    loss_or = model.loss(pred_or, s)

    train_loss = loss_sr + loss_or
    
    if (args.reg_weight != 0.0 and args.reg_norm == 3):
        #self.logger.info('Computing regularizer weight')
        if model == 'complex':
            emb_dim = args.embedding_dim #int(self.args.embedding_dim/2)
            lhs = (emb_s[:, :emb_dim], emb_s[:, emb_dim:])
            rel = (emb_r[:, :emb_dim], emb_r[:, emb_dim:])
            rel_rev = (emb_rrev[:, :emb_dim], emb_rrev[:, emb_dim:])
            rhs = (emb_o[:, :emb_dim], emb_o[:, emb_dim:])

            #print(lhs[0].shape, lhs[1].shape)
            factors_sr = (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                      )
            factors_or = (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel_rev[0] ** 2 + rel_rev[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                      )
        else:
            factors_sr = (emb_s, emb_r, emb_o)
            factors_or = (emb_s, emb_rrev, emb_o)

        train_loss  += n3_regularizer(factors_sr, args.reg_weight, p=3)
        train_loss  += n3_regularizer(factors_or, args.reg_weight, p=3)

    if (args.reg_weight != 0.0 and args.reg_norm == 2):
        train_loss += lp_regularizer(model, args.reg_weight, p=2)
    
    return train_loss

def hv(loss, model_params, v): # from pytorch issue #24004
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    #print(Hv)
    return Hv

def get_inverse_hvp_lissa(v, model, device, param_influence, train_data, args):
    damping = args.damping
    num_samples = args.lissa_repeat
    scale = args.scale #default = 1e-4
    train_batch_size = args.lissa_batch_size
    lissa_num_batches = math.ceil(train_data.shape[0]/train_batch_size)
    recursion_depth = int(lissa_num_batches*args.lissa_depth)

    
    ihvp = None
    for i in range(num_samples):
        cur_estimate = v
        #lissa_data_iterator = iter(train_loader)
        input_data = torch.from_numpy(train_data.astype('int64'))
        actual_examples = input_data[torch.randperm(input_data.shape[0]), :]
        del input_data
        
        b_begin = 0
        for j in range(recursion_depth):
            model.zero_grad() # same as optimizer.zero_grad()
            if b_begin >= actual_examples.shape[0]:
                b_begin = 0
                input_data = torch.from_numpy(train_data.astype('int64'))
                actual_examples = input_data[torch.randperm(input_data.shape[0]), :]
                del input_data
            
            input_batch = actual_examples[b_begin: b_begin + train_batch_size]
            input_batch = input_batch.to(device)
            
            train_loss = get_train_loss(input_batch, model, device, args)
            
            hvp = hv(train_loss, param_influence, cur_estimate)
            cur_estimate = [_a + (1-damping)*_b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
            if (j%200 == 0) or (j == recursion_depth -1 ):
                logger.info("Recursion at depth %s: norm is %f" % (j, np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy())))
            
            b_begin += train_batch_size
        
        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]

    logger.info("Final ihvp norm is %f" % (np.linalg.norm(gather_flat_grad(ihvp).cpu().numpy())))
    #print(len(ihvp), ihvp[0].shape, ihvp[1].shape) #there's one ihvp for every param matrix
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= num_samples
    
    return return_ihvp


def get_deletions(train_data, test_data, neighbours, model, args):
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))

    triples_to_delete = []
    if_scores = {}
    for test_idx, test_trip in enumerate(test_data):
        test_nghbrs = neighbours[test_idx]
        nghbr_trip = train_data[test_nghbrs]
        test_trip = test_trip[None, :] # add a batch dimension
        test_trip = torch.from_numpy(test_trip).to(device)
        #test_s, test_r, test_o = test_trip[:,0], test_trip[:,1], test_trip[:,2]
        #test_vec = model.score_triples_vec(test_s, test_r, test_o)

        #### L-test gradient ####
        model.eval()
        model.zero_grad()
        test_loss = get_model_loss(test_trip, model, device)
        test_grads = autograd.grad(test_loss, param_influence)
        #test_grads = gather_flat_grad(test_grads)
        
        ##### IHVP #######
        model.train()
        inverse_hvp = get_inverse_hvp_lissa(test_grads, model, device, 
                                            param_influence, train_data, args)

        model.eval()
        influences = []
        for train_trip in nghbr_trip:
            #model.train() #batch norm cannot be used here
            train_trip = train_trip[None, :] # add batch dim
            train_trip = torch.from_numpy(train_trip).to(device)
            #### L-train gradient ####
            model.zero_grad()
            train_loss = get_model_loss(train_trip, model, device)
            train_grads = autograd.grad(train_loss, param_influence)
            train_grads = gather_flat_grad(train_grads)
            influence = torch.dot(inverse_hvp, train_grads) #default dim=1
            influences.append(influence.item())  

        influences = np.array(influences)
        influences = torch.from_numpy(influences).to(device)
        # we want to remove the neighbour with maximum influence
        max_values, argsort = torch.sort(influences, -1, descending=True)
        del_idx = argsort[0]
        triple_to_delete = nghbr_trip[del_idx]

        triples_to_delete.append(triple_to_delete)
        
        if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
            logger.info('--------- Processed test triple {0} -----------'.format(str(test_idx)))
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
    
    args.seed = args.seed + (args.rand_run - 1) # default seed is 17
    
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
    log_path = 'logs/attack_logs/if_del_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
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
    
    param_optimizer = list(model.named_parameters())
    param_influence = []
    for n,p in param_optimizer:
        param_influence.append(p)
        
    neighbours = utils.generate_nghbrs(test_data, train_data) 
    # test set is the target set because we loaded data from target_...
    
    if args.reproduce_results:
        args = utils.set_if_params(args)
    else:
        args.damping = 0.01 
        # damping is needed to keep the hessian matrix PSD; it is the most negative eigenvalue of empirical hessian
        args.lissa_repeat = 1 
        args.lissa_depth = 1
        args.scale = 500
        args.lissa_batch_size = 100
        
    # use batches to select recursion depth
    lissa_num_batches = math.ceil(train_data.shape[0]/args.lissa_batch_size)
    
    logger.info('-------- Lissa Params for IHVP --------')
    logger.info('Damping: {0}'.format(args.damping))
    logger.info('Lissa_repeat: {0}'.format(args.lissa_repeat))
    logger.info('Lissa_depth: {0}'.format(args.lissa_depth))
    logger.info('Scale: {0}'.format(args.scale))
    logger.info('Lissa batch size: {0}'.format(args.lissa_batch_size))
    logger.info('Lissa num bacthes: {0}'.format(lissa_num_batches))
    
    
    triples_to_delete = get_deletions(train_data, test_data, neighbours, 
                                  model, args)
    
    df = pd.DataFrame(data=triples_to_delete)
    df = df.drop_duplicates()
    #print(df.shape)
    trips_to_delete = df.values
    #print(trips_to_delete.shape)
    num_duplicates = len(triples_to_delete) - trips_to_delete.shape[0]
    #print(num_duplicates)

    per_tr_1, n_ignored_edits = utils.perturb_data(train_data, 
                                               trips_to_delete)
    
    # Perturbed dataset
    logger.info('Shape of perturbed training set: {0}'.format(per_tr_1.shape))
    logger.info('Number of adversarial deletions ignored (because of singleton nodes): {0}'.format(n_ignored_edits))
    logger.info('Number of duplicate adversarial deletions : {0}'.format(num_duplicates))


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(per_tr_1.shape[0]))


    save_path = 'data/if_del_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
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


    # In[ ]:





    # In[ ]:







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


