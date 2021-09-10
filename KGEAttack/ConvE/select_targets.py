import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import json
import logging
import argparse 
import math
from pprint import pprint

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe
import utils


def set_paths(args):
    model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    
    log_path = 'logs/select_target_{0}_{1}_{2}_{3}_{4}.log'.format(args.data, args.target_split, model_name, args.epochs, args.train_batch_size)
    eval_name = '{0}_{1}_{2}_{3}_{4}_{5}'.format(args.data, model_name, args.epochs, args.train_batch_size, args.valid_batch_size, args.test_batch_size)
    
    return model_name, model_path, eval_name, log_path


def get_ranking(model, queries:torch.Tensor, num_rel:int,
               filters:Dict[str, Dict[Tuple[int, int], List[int]]],
                device: str,
                batch_size: int = 500
               ):
    ranks = []
    ranks_lhs = []
    ranks_rhs = []
    b_begin = 0
    #logger.info('Computing ranks for all queries')
    while b_begin < len(queries):
        b_queries = queries[b_begin : b_begin+batch_size]
        s,r,o = b_queries[:,0], b_queries[:,1], b_queries[:,2]
        r_rev = r+num_rel
        lhs_score = model.score_or(o,r_rev, sigmoid=False) #this gives scores not probabilities
        rhs_score = model.score_sr(s,r,sigmoid=False) # this gives scores not probabilities

        for i, query in enumerate(b_queries):
            filter_lhs = filters['lhs'][(query[2].item(), query[1].item())]
            filter_rhs = filters['rhs'][(query[0].item(), query[1].item())]

            # save the prediction that is relevant
            target_value1 = rhs_score[i, query[2].item()].item()
            target_value2 = lhs_score[i, query[0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            lhs_score[i][filter_lhs] = -1e6
            rhs_score[i][filter_rhs] = -1e6
            # write base the saved values
            rhs_score[i][query[2].item()] = target_value1
            lhs_score[i][query[0].item()] = target_value2

        # sort and rank
        max_values, lhs_sort = torch.sort(lhs_score, dim=1, descending=True) #high scores get low number ranks
        max_values, rhs_sort = torch.sort(rhs_score, dim=1, descending=True)

        lhs_sort = lhs_sort.cpu().numpy()
        rhs_sort = rhs_sort.cpu().numpy()

        for i, query in enumerate(b_queries):
            # find the rank of the target entities
            lhs_rank = np.where(lhs_sort[i]==query[0].item())[0][0]
            rhs_rank = np.where(rhs_sort[i]==query[2].item())[0][0]

            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks_lhs.append(lhs_rank + 1)
            ranks_rhs.append(rhs_rank + 1)

        b_begin += batch_size

    #logger.info('Ranking done for all queries')
    return ranks_lhs, ranks_rhs 
   
    

if __name__ == '__main__':
    parser = utils.get_argument_parser()
    parser.add_argument('--target-split', type=int, default=0, help='Ranks to use for target set. Values are 0 for ranks ==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    args = parser.parse_args()
    if args.reproduce_results:
        args = utils.set_hyperparams(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fixing random seeds for reproducibility -https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)
    
    args.epochs = -1 #no training here
    model_name, model_path, eval_name, log_path = set_paths(args)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
    logger = logging.getLogger(__name__)
    
    data_path = 'data/{0}'.format(args.data)
    n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)
    
    ##### load data####
    data  = utils.load_data(data_path)
    train_data, valid_data, test_data = data['train'], data['valid'], data['test']
    
    inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
    to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
    inp_f.close()
    to_skip_eval['lhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['lhs'].items()}
    to_skip_eval['rhs'] = {(int(k[0]), int(k[1])): v for k,v in to_skip_eval['rhs'].items()}
    
    # add a model and load the pre-trained params
    model = utils.load_model(model_path, args, n_ent, n_rel, device)
    
    with torch.no_grad():
        target_path = 'data/target_{0}_{1}_{2}'.format(args.model, args.data, args.target_split)
        
        # generate ranks for test set 
        logger.info('Generating target set from test set')
        test_data = torch.from_numpy(test_data.astype('int64')).to(device)
        if args.add_reciprocals:
            num_rel= n_rel
        else:
            num_rel = 0
        ranks_lhs, ranks_rhs = get_ranking(model, test_data, num_rel, to_skip_eval, device, args.test_batch_size)
        ranks_lhs, ranks_rhs = np.array(ranks_lhs), np.array(ranks_rhs)
        #indices_lhs, indices_rhs = np.asarray(ranks_lhs <= 10).nonzero(), np.asarray(ranks_rhs <= 10).nonzero()
        if args.target_split == 2:
            indices = np.asarray(((ranks_lhs <= 100) & (ranks_lhs >10)) & ((ranks_rhs <= 100)&(ranks_rhs > 10))).nonzero()
        elif args.target_split ==1 :
            indices = np.asarray((ranks_lhs <= 10) & (ranks_rhs <= 10)).nonzero()
        elif args.target_split ==0 :
            indices = np.asarray((ranks_lhs == 1) & (ranks_rhs == 1)).nonzero()
        else:
            logger.info('Unknown Target Split: {0}', args.target_split)
            raise Exception("Unknown target split!")
        
        test_data = test_data.cpu().numpy()
        #targets_lhs, targets_rhs = test_data[indices_lhs], test_data[indices_rhs]
        targets = test_data[indices]
        logger.info('Number of targets generated: {0}'.format(targets.shape[0]))
        #save eval for selected targets
        split = 'target_{0}'.format(args.target_split)
        
        results_target = evaluation(model, targets, to_skip_eval, eval_name, num_rel, split, args.test_batch_size, -1, device)
        # save target set

        with open(os.path.join(target_path, 'target.txt'), 'w') as out:
            for item in targets:
                out.write("%s\n" % "\t".join(map(str, item)))
        with open(os.path.join(target_path, 'test.txt'), 'w') as out:
            for item in targets:
                out.write("%s\n" % "\t".join(map(str, item)))

        # use the valid set to generate non-target set
        logger.info('Generating non target set from valid set')
        valid_data = torch.from_numpy(valid_data.astype('int64')).to(device)
        if args.add_reciprocals:
            num_rel= n_rel
        else:
            num_rel = 0
        ranks_lhs, ranks_rhs = get_ranking(model, valid_data, num_rel, to_skip_eval, device, args.valid_batch_size)
        ranks_lhs, ranks_rhs = np.array(ranks_lhs), np.array(ranks_rhs)
        if args.target_split == 2:
            indices = np.asarray(((ranks_lhs <= 100) & (ranks_lhs >10)) & ((ranks_rhs <= 100)&(ranks_rhs > 10))).nonzero()
        elif args.target_split == 1:
            indices = np.asarray((ranks_lhs <= 10) & (ranks_rhs <= 10)).nonzero()
        elif args.target_split ==0 :
            indices = np.asarray((ranks_lhs == 1) & (ranks_rhs == 1)).nonzero()
        else:
            logger.info('Unknown Target Split: {0}', self.args.target_split)
            raise Exception("Unknown target split!")
            
        valid_data = valid_data.cpu().numpy()
        non_targets = valid_data[indices]
        logger.info('Number of non targets generated: {0}'.format(non_targets.shape[0]))
        #save eval for selected non targets
        split = 'non_target_{0}'.format(args.target_split)
        
        results_ntarget = evaluation(model, non_targets, to_skip_eval, eval_name, num_rel, split, args.valid_batch_size, -1, device)
        # save non target set and valid set both - eval needed for both
        with open(os.path.join(target_path, 'non_target.txt'), 'w') as out:
            for item in non_targets:
                out.write("%s\n" % "\t".join(map(str, item)))
        with open(os.path.join(target_path, 'valid.txt'), 'w') as out:
            for item in valid_data:
                out.write("%s\n" % "\t".join(map(str, item)))


        # saving dicts to avoid searching later
        with open(os.path.join(target_path, 'entities_dict.json'), 'w') as f:
            f.write(json.dumps(ent_to_id)  + '\n')

        with open(os.path.join(target_path, 'relations_dict.json'), 'w') as f:
            f.write(json.dumps(rel_to_id)  + '\n')
            
        with open(os.path.join(target_path, 'train.txt'), 'w') as out:
            for item in train_data:
                out.write("%s\n" % "\t".join(map(str, item)))
                
        out = open(os.path.join(target_path, 'to_skip_eval.pickle'), 'wb')
        pickle.dump(to_skip_eval, out)
        out.close()

        # write down the stats for targets generated
        with open(os.path.join(target_path, 'stats.txt'), 'w') as out:
            out.write('Number of train set triples: {0}\n'.format(train_data.shape[0]))
            out.write('Number of test set triples: {0}\n'.format(test_data.shape[0]))
            out.write('Number of valid set triples: {0}\n'.format(valid_data.shape[0]))
            out.write('Number of target triples: {0}\n'.format(targets.shape[0]))
            out.write('Number of non target triples: {0}\n'.format(non_targets.shape[0]))
            if args.target_split ==2:
                out.write('Target triples are ranked >10 and <=100 and test set is the target triples \n')
                out.write('Non target triples are ranked >10 and <=100 but valid triples is original valid set \n')
                out.write('Non target triples with ranks >10 and <=100 are in non_target.txt \n')
            elif args.target_split == 0:
                out.write('Target triples are ranked =1 and test set is the target triples \n')
                out.write('Non target triples are ranked =1 but valid triples is original valid set \n')
                out.write('Non target triples with ranks =1 are in non_target.txt \n')
            else:
                out.write('Target triples are ranked <=10 and test set is the target triples \n')
                out.write('Non target triples are ranked <=10 but valid triples is original valid set \n')
                out.write('Non target triples with ranks <=10 are in non_target.txt \n')
            out.write('------------------------------------------- \n')
    
    
