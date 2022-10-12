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

from sklearn.cluster import MiniBatchKMeans, KMeans
import time

import torch
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch.autograd as autograd

from model import Distmult, Complex, Conve, Transe
import utils


# - In this notebook - 
#     - generate clusters for model, data combinations
#     - save them
#     
#  


if __name__ == '__main__':


    parser = utils.get_argument_parser()
    parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')
    parser.add_argument('--rand-run', type=int, default=1, help='A number assigned to the random run of experiment')
    parser.add_argument('--num-clusters', type=int, default=100, help='Number of clusters to be generated')


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
    #log_path = 'logs/attack_logs/com_add_2_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
    #                                                           args.target_split, args.budget, args.rand_run)
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO
                           )
    logger = logging.getLogger(__name__)
    logger.info('-------------------- Edits with Composition Attack - worse ranks ----------------------')
    
    data_path = 'data/target_{0}_{1}_{2}'.format(args.model, args.data, args.target_split)

    n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)



    model = utils.load_model(model_path, args, n_ent, n_rel, device)
    
    logger.info("Starting the clustering algorithm")
    # Perform clustering of entity embeddings
    ent_emb = model.emb_e.weight.data
    ent_emb = ent_emb.cpu().numpy()

    km = KMeans(n_clusters=args.num_clusters, n_init=100, max_iter=500, 
                             random_state=0, #batch_size = 100, 
                             init='k-means++'#, verbose=1
                             #max_no_improvement=20
                            )
    km.fit(ent_emb)
    
    logger.info("Finished clustering... saving centres, labels, inertia, n_iter")
    
    save_path = 'clusters/{0}_{1}_{2}_{3}'.format( args.model, args.data, args.num_clusters, args.rand_run)
    
    out = open(save_path + 'cluster_centers.pickle', 'wb')
    pickle.dump(km.cluster_centers_, out)
    out.close()
    
    out = open(save_path + 'labels.pickle', 'wb')
    pickle.dump(km.labels_, out)
    out.close()
    
    out = open(save_path + 'inertia.pickle', 'wb')
    pickle.dump(km.inertia_, out)
    out.close()
    
    out = open(save_path + 'n_iter.pickle', 'wb')
    pickle.dump(km.n_iter_, out)
    out.close()
    
    
    
    
