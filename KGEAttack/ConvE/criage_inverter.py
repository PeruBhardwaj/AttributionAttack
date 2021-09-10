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
from criage_model import Distmult, Conve
import utils

def add_model(args, n_ent, n_rel):
    if args.add_reciprocals:
        if args.model is None:
            model = Distmult(args, n_ent, 2*n_rel)
        elif args.model == 'conve':
            model = Conve(args, n_ent, 2*n_rel)
        elif args.model == 'distmult':
            model = Distmult(args, n_ent, 2*n_rel)
        else:
            logger.info('Unknown model: {0}', args.model)
            raise Exception("Unknown model!")
    else:
        if args.model is None:
            model = Distmult(args, n_ent, n_rel)
        elif args.model == 'conve':
            model = Conve(args, n_ent, n_rel)
        elif args.model == 'distmult':
            model = Distmult(args, n_ent, n_rel)
        else:
            logger.info('Unknown model: {0}', args.model)
            raise Exception("Unknown model!")

    #model.to(self.device)
    return model


if __name__ == '__main__':


    parser = utils.get_argument_parser()
    parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
    parser.add_argument('--attack-batch-size', type=int, default=1000, help='Batch size for processing neighbours of target')

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
    log_path = 'logs/attack_logs/criage_inverter_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                               args.target_split, args.budget, args.rand_run)


    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info('-------------------- Running Criage Inverter ----------------------')
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
    
    
    logger.info('Loading pre-trained model params')
    # add a model and load the pre-trained params
    model = add_model(args, n_ent, n_rel)
    model.to(device)
    logger.info('Loading saved model from {0}'.format(model_path))
    model_state = model.state_dict()
    pre_state = torch.load(model_path)
    pretrained = pre_state['state_dict']
    for name in model_state:
        if name in pretrained:
            model_state[name].copy_(pretrained[name])
            
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    
    logger.info('----- Training -----')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        
        #shuffle the train dataset
        input_data = torch.from_numpy(train_data.astype('int64'))
        actual_examples = input_data[torch.randperm(input_data.shape[0]), :]
        del input_data
        
        batch_size = args.train_batch_size
        b_begin = 0
        
        while b_begin < actual_examples.shape[0]:
            optimizer.zero_grad()
            input_batch = actual_examples[b_begin: b_begin + batch_size]
            input_batch = input_batch.to(self.device)
            
            e1,rel,e2 = input_batch[:,0], input_batch[:,1], input_batch[:,2]
                
            E1, R = model.forward(e1, rel)
            loss_E1 = model.loss(E1, e1) #e1.squeeze(1))
            loss_R = model.loss(R, rel) #rel.squeeze(1))
            loss = loss_E1 + loss_R
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (b%100 == 0) or (b== (2*args.num_batches-1)):
                logger.info('[E:{} | {}]: Train Loss:{:.4}'.format(epoch, b, np.mean(losses)))
                
            b_begin += batch_size
            
        loss = np.mean(losses)
        logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
            
    
    logger.info('Saving trained inverter model')
    save_path = 'saved_models/criage_inverter/{0}_{1}.model'.format(args.data, model_name)
    state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args)
            }
    torch.save(state, save_path)
    logger.info('Saving model to {0}'.format(save_path))
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
