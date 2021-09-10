## This is the notebook for randomly selecting k target triples from the target test set. 

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


from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe
import utils

if __name__ == '__main__':
    parser = utils.get_argument_parser()
    parser.add_argument('--target-split', type=int, default=0, help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
    parser.add_argument('--num-targets', type=int, default=100, help='Number of target triples to select randomly')
    
    args = parser.parse_args()
    if args.reproduce_results:
        args = utils.set_hyperparams(args)

    #args.target_split = 0 # which target split to use 
    #Values are 1 for ranks <=10; 2 for ranks>10 and ranks<=100.
    #args.budget = 1 #indicates the num of adversarial edits for each target triple for each corruption side
    #args.rand_run = 1 #  a number assigned to the random run of the experiment
    args.seed = args.seed + (args.rand_run - 1) # default seed is 17
    #args.num_targets = 100

    #args.data = 'WN18RR'
    #args.model = 'distmult'
    #k = args.num_targets # number of target triples to select randomly
    new_target_split_save = '{0}_{1}_{2}'.format(args.target_split, 
                                                 args.num_targets, 
                                                 args.rand_run)
    # Eg - if original split was 0, then new split is 0_100_1
    
    
    # Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
    # torch.manual_seed(args.seed)
    # cudnn.deterministic = True
    # cudnn.benchmark = False
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)


    model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
    log_path = 'logs/select_rand_target_{0}_{1}_{2}_{3}_{4}.log'.format(args.data, args.target_split, model_name, args.epochs, args.train_batch_size)
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
    
    
    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of target test set: ' + str(test_data.shape[0]))
    
    
    if test_data.shape[0] > args.num_targets:
        # need to make sure that duplicate target triples are not selected
        remaining_targets = np.empty_like(test_data)
        remaining_targets = test_data

        targets_selected = []
        for idx in range(args.num_targets):
            if (idx%100 == 0 or idx == args.num_targets-1):
                logger.info('Selecting random triple ' + str(idx))

            # select a random index from available targets
            target_idx = rng.choice(a=remaining_targets.shape[0], size = 1, replace=False)[0]
            # select the target triple at this idx
            target_trip = remaining_targets[target_idx]
            targets_selected.append(target_trip)

            # update the available targets to avoid duplicates
            m = np.ones(remaining_targets.shape[0], bool)
            m[target_idx] = False
            remaining_targets = remaining_targets[m]


        targets_selected = np.array(targets_selected)
    else:
        targets_selected = test_data
    
    logger.info ('Length of original target set: ' + str(test_data.shape[0]))
    logger.info ('Length of target subset selected randomly: ' + str(targets_selected.shape[0]))
    
    save_path = 'data/target_{0}_{1}_{2}'.format(args.model, args.data, new_target_split_save)
    
    try :
        os.makedirs(save_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            logger.info(e)
            logger.info('Using the existing folder {0} for processed data'.format(save_path))
        else:
            raise
            
    with open(os.path.join(save_path, 'train.txt'), 'w') as out:
        for item in train_data:
            out.write("%s\n" % "\t".join(map(str, item)))
            
    with open(os.path.join(save_path, 'entities_dict.json'), 'w') as f:
        f.write(json.dumps(ent_to_id)  + '\n')

    with open(os.path.join(save_path, 'relations_dict.json'), 'w') as f:
        f.write(json.dumps(rel_to_id)  + '\n')
        
    with open(os.path.join(save_path, 'valid.txt'), 'w') as out:
        for item in valid_data:
            out.write("%s\n" % "\t".join(map(str, item)))

    # out = open(os.path.join(save_path, 'valid.pickle'), 'wb')
    # pickle.dump(valid_data.astype('uint64'), out)
    # out.close()
    
    out = open(os.path.join(save_path, 'to_skip_eval.pickle'), 'wb')
    pickle.dump(to_skip_eval, out)
    out.close()
    
    with open(os.path.join(save_path, 'test.txt'), 'w') as out:
        for item in targets_selected:
            out.write("%s\n" % "\t".join(map(str, item)))
        
    # out = open(os.path.join(save_path, 'test.pickle'), 'wb')
    # pickle.dump(targets_selected.astype('uint64'), out)
    # out.close()
    
    
    with open(os.path.join(save_path, 'original_target.txt'), 'w') as out:
        for item in test_data:
            out.write("%s\n" % "\t".join(map(str, item)))
            
    df = pd.read_csv(os.path.join(data_path, 'non_target'+'.txt'), sep='\t', header=None, names=None, dtype=int)
    df = df.drop_duplicates()
    non_targets= df.values
    with open(os.path.join(save_path, 'non_target.txt'), 'w') as out:
        for item in non_targets:
            out.write("%s\n" % "\t".join(map(str, item)))
            
            
    with open(os.path.join(save_path, 'stats.txt'), 'w') as f:
        f.write('Length of original target set: {0}\n'.format(test_data.shape[0]))
        f.write('Number of target triples selected: {0}\n'.format(targets_selected.shape[0]))
        f.write('Length of training set: {0} \n'. format(train_data.shape[0]))
        f.write('Length of valid set: {0} \n'. format(valid_data.shape[0]))
        f.write('Number of non target triples: {0}\n'.format(non_targets.shape[0]))
        if args.target_split == 0:
            f.write('Target triples are ranked =1 and test set is the subset of target triples \n')
            f.write('Non target triples are ranked =1 but valid triples is original valid set \n')
            f.write('Non target triples with ranks =1 are in non_target.txt \n')
        else:
            f.write('Target triples are ranked <=10 and test set is subset of the target triples \n')
            f.write('Non target triples are ranked <=10 but valid triples is original valid set \n')
            f.write('Non target triples with ranks <=10 are in non_target.txt \n')
        f.write('---------------------------------------------------------------------- \n')



