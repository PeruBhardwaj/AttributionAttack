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

def load_model(model_path, args, n_ent, n_rel, device):
    # add a model and load the pre-trained params
    model = add_model(args, n_ent, n_rel)
    model.to(device)
    logger.info('Loading saved model from {0}'.format(model_path))
    state = torch.load(model_path)
    model_params = state['state_dict']
    params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
    for key, size, count in params:
        logger.info('Key:{0}, Size:{1}, Count:{2}'.format(key, size, count))
        
    model.load_state_dict(model_params)
    model.eval()
    logger.info(model)
    
    return model


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
    Score = np.dot(Y, np.transpose((1-sig_tri)*np.dot(X, M)))
    return Score, M

def grad_score(Y, X, H, e_o, M):
    grad = []
    n = 200
    sig_tri = sig(e_o, X)
    A = H + (sig_tri)*(1-sig_tri)*np.dot(np.transpose(X), X)
    A_in = M 
    X_2 = np.dot(np.transpose(X), X)
    f_part = np.dot(Y, np.dot((1-sig_tri)*np.eye(n)-(sig_tri)*(1-sig_tri)*np.transpose(np.dot(np.transpose(e_o), X)), A_in))
    for i in range(n):
        s = np.zeros((n,n))
        s[:,i] = X
        s[i,:] = X
        s[i,i] = 2*X[0][i]
        Q = np.dot(((sig_tri)*(1-sig_tri)**2-(sig_tri)**2*(1-sig_tri))*e_o[0][i]*X_2+(sig_tri)*(1-sig_tri)*s, A_in)
        grad += [f_part[0][i] - np.dot(Y, np.transpose((1-sig_tri)*np.dot(X, np.dot(A_in, Q))))[0][0]] ######## + 0.02 * X[0][i]]

    return grad

def find_best_attack(e_o, Y, nei, embd_e, embd_rel, pr):
    H = point_hess(e_o, nei, embd_e, embd_rel)
    X = pr
    step = np.array([[0.00000000001]])
    score = 0 
    score_orig,_ = point_score(Y, pr, e_o,H)
    score_n, M = point_score(Y, X, e_o,H)
    num_iter = 0
    
    atk_flag = 0
    while score_n >= score_orig or num_iter<1:
        if num_iter ==4:
            X = pr
            atk_flag = 1
            print('Returning from find_best_attack without update')
            break
        num_iter += 1
        Grad = grad_score(Y, X, H, e_o, M)
        X = X + step * Grad 
        score = score_n
        score_n, M = point_score(Y, X, e_o, H)

    return X, atk_flag

def find_best_at(pred, E2):
    e2 = E2.view(-1).data.cpu().numpy()
    Pred = pred.view(-1).data.cpu().numpy()
    A1 = np.dot(Pred, e2)
    A2 = np.dot(e2, e2)
    A3 = np.dot(Pred, Pred)
    # I am adding this because I got a math domain error for sqrt (distmult nations)
    a = np.true_divide(A3*A2-0.2, A3*A2-A1**2)
    if a>0 :
        A = math.sqrt(np.true_divide(A3*A2-0.2, A3*A2-A1**2))
    else:
        A = 0
    #A = math.sqrt(np.true_divide(A3*A2-0.2, A3*A2-A1**2))
    B = np.true_divide(math.sqrt(0.2)-A*A1, A2)  
    return float(A), float(B)


def get_nghbr_o_addition(train_data, test_trip, model):
    
    logger.info("----------------- Generating triples of type s'r'o -----------------")
    test_trip = test_trip[None, :] # add a batch dimension
    test_trip = torch.from_numpy(test_trip).to(device)
    e1,rel,e2 = test_trip[:,0], test_trip[:,1], test_trip[:,2]
    
    pred = model.encoder(e1, rel)
    E2 = model.encoder_2(e2)
    
    A, B = find_best_at(-pred, E2)
    attack_ext = -A*pred+B*E2
    # Note - the code on GitHub uses find_best_at instead of find_best_attack
    
    #if e2_or in E2_dict:
    #    nei = E2_dict[e2_or]
    #    attack, flag = find_best_attack(E2.data.cpu().numpy(), pred.data.cpu().numpy(), nei, embd_e, embd_rel, attack_ext.cpu().detach().numpy())
    #    attack = torch.autograd.Variable(torch.from_numpy(attack)).cuda().float()
        #attack = attack_ext
    #    no_atk_found += flag # flag is 1 when grad update does not happen, 0 otherwise

    #else: 
    #    print('Gradient attack not found for triple number: ', n_t) #this excludes the break inside the function
    #    no_atk_found += 1
    #    attack = attack_ext
    attack = attack_ext
    
    E1, R = model.decoder(attack)
    _, predicted_e1 = torch.max(E1, 1)
    _, predicted_R = torch.max(R, 1)
    
    trip_to_add = [predicted_e1.item(), predicted_R.item(), e2.item()]
    
    # compute criage score of the predicted triple (to choose between s and o additions)
    e_o = E2.data.cpu().numpy()
    Y1 = pred.data.cpu().numpy()
    nei1 = [trip_to_add]
    emb_e = model.emb_e.weight.data.cpu().numpy()
    emb_rel = model.emb_rel.weight.data.cpu().numpy()
    e1_or = trip_to_add[0]
    rel = trip_to_add[1]
    
    H1 = point_hess(e_o, nei1, emb_e, emb_rel)
    e1 = torch.cuda.LongTensor([e1_or])
    rel = torch.cuda.LongTensor([rel])
    pred = model.encoder(e1, rel).data.cpu().numpy()
    score_t, M = point_score(Y1, pred, e_o , H1)
    score = score_t[0][0]
    
    return trip_to_add, score


def get_nghbr_s_addition(train_data, test_trip, model):
    
    logger.info("----------------- Generating triples of type sr'o' ------------------------")
    test_trip = test_trip[None, :] # add a batch dimension
    test_trip = torch.from_numpy(test_trip).to(device)
    e1,rel,e2 = test_trip[:,0], test_trip[:,1], test_trip[:,2]
    
    pred = model.encoder(e2, rel)
    E1 = model.encoder_2(e1) 
    
    A, B = find_best_at(-pred, E1)
    attack_ext = -A*pred+B*E1
    #if e1_or in E1_dict:
    #    nei = E1_dict[e1_or]
    #    attack, flag = find_best_attack(E1.data.cpu().numpy(), pred.data.cpu().numpy(), nei, embd_e, embd_rel, attack_ext.cpu().detach().numpy())
    #    attack = torch.autograd.Variable(torch.from_numpy(attack)).cuda().float()
        #attack = attack_ext
    #    no_atk_found += flag # flag is 1 when grad update does not happen, 0 otherwise

    #else: 
    #    print('Gradient attack not found for triple number: ', n_t) #this excludes the break inside the function
    #    no_atk_found += 1
    #    attack = attack_ext
    
    attack = attack_ext
    
    E2, R = model.decoder(attack)
    _, predicted_e2 = torch.max(E2, 1)
    _, predicted_R = torch.max(R, 1)
    
    trip_to_add = [e1.item(), predicted_R.item(), predicted_e2.item()]
    
    # compute criage score of the predicted triple (to choose between s and o additions)
    e_s = E1.data.cpu().numpy()
    Y2 = pred.data.cpu().numpy()
    nei2 = [trip_to_add]
    emb_e = model.emb_e.weight.data.cpu().numpy()
    emb_rel = model.emb_rel.weight.data.cpu().numpy()
    e2_or = trip_to_add[2]
    rel = trip_to_add[1]
    
    H2 = point_hess(e_s, nei2, emb_e, emb_rel)
    e2 = torch.cuda.LongTensor([e2_or])
    rel = torch.cuda.LongTensor([rel])
    pred = model.encoder(e2, rel).data.cpu().numpy()
    score_t, M = point_score(Y2, pred, e_s , H2)
    score = score_t[0][0]
    
    return trip_to_add, score

def get_additions(train_data, test_data, model):
    
    
    logger.info('------ Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(start_time))
    
    triples_to_add = []
    for test_idx, test_trip in enumerate(test_data):
        
        add_trip_s, max_val_s = get_nghbr_s_addition(train_data, test_trip, model)
        add_trip_o, max_val_o = get_nghbr_o_addition(train_data, test_trip, model)

        if max_val_s > max_val_o:
            add_trip = add_trip_s
        else:
            add_trip = add_trip_o
    
        
        #triple_to_add = add_trip

        triples_to_add.append(add_trip)
        if test_idx%100 == 0 or test_idx == test_data.shape[0]-1:
            logger.info('Processed test triple {0}'.format(test_idx))
            logger.info('Time taken: {0}'.format(time.time() - start_time))
    logger.info('Time taken to generate edits: {0}'.format(time.time() - start_time))    


    
    return np.array(triples_to_add)


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
    log_path = 'logs/attack_logs/criage_add_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
                                                               args.target_split, args.budget, args.rand_run)
    
    logger = logging.getLogger(__name__)
    logger.info('-------------------- Edits with Criage baseline ----------------------')
    logger.info(args)
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
    
    logger.info('Loading pre-trained inverter model')
    inverter_model_path = 'saved_models/criage_inverter/{0}_{1}.model'.format(args.data, model_name)
    model = load_model(inverter_model_path, args, n_ent, n_rel, device)
    
    triples_to_add = get_additions(train_data, test_data, model)
    
    #remove duplicate entries
    df = pd.DataFrame(data=triples_to_add)
    df = df.drop_duplicates()
    # print(df.shape)
    trips_to_add = df.values
    # print(trips_to_delete.shape)
    num_duplicates = len(triples_to_add) - trips_to_add.shape[0]
    # print(num_duplicates)
    
    per_tr = np.concatenate((trips_to_add, train_data))
    
    #remove duplicate entries
    df = pd.DataFrame(data=per_tr)
    df = df.drop_duplicates()
    # print(df.shape)
    per_tr_1 = df.values
    # print(trips_to_delete.shape)
    num_duplicates_1 = per_tr.shape[0] - per_tr_1.shape[0]
    # print(num_duplicates)
    
    logger.info('Shape of perturbed training set: {0}'.format(per_tr_1.shape))
    logger.info('Number of duplicate adversarial additions: {0}'.format(num_duplicates))
    logger.info('Number of adversarial additions already in train data: {0}'.format(num_duplicates_1))


    logger.info ('Length of original training set: ' + str(train_data.shape[0]))
    logger.info ('Length of new poisoned training set: ' + str(per_tr_1.shape[0]))


    save_path = 'data/criage_add_{0}_{1}_{2}_{3}_{4}'.format( args.model, args.data, 
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


    
























