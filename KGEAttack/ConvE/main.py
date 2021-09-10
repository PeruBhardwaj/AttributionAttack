import pickle
from typing import Dict, Tuple, List
import os
import numpy as np
import json
import logging
import argparse 
import math
from pprint import pprint
import pandas as pd
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.autograd as autograd

from evaluation import evaluation
from model import Distmult, Complex, Conve, Transe
import utils

class Main(object):
    def __init__(self, args):
        self.args = args 
        
        self.model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
        #leaving batches from the model_name since they do not depend on model_architecture 
        # also leaving kernel size and filters, siinice don't intend to change those
        self.model_path = 'saved_models/{0}_{1}.model'.format(args.data, self.model_name)
        
        self.log_path = 'logs/{0}_{1}_{2}_{3}.log'.format(args.data, self.model_name, args.epochs, args.train_batch_size)
        self.eval_name = '{0}_{1}_{2}_{3}_{4}_{5}'.format(args.data, self.model_name, args.epochs, args.train_batch_size, args.valid_batch_size, args.test_batch_size)
        self.loss_path = 'losses/{0}_{1}_{2}_{3}.pickle'.format(args.data, self.model_name, args.epochs, args.train_batch_size)
        
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = self.log_path
                           )
        self.logger = logging.getLogger(__name__)
        self.logger.info(vars(self.args))
        
        if self.args.save_influence_map:
            # when we want to save influence during training
            self.args.add_reciprocals = False # to keep things simple
            # init an empty influence map
            self.influence_map = defaultdict(float)
            #self.influence_path = 'influence_maps/{0}_{1}.json'.format(args.data, self.model_name)
            self.influence_path = 'influence_maps/{0}_{1}.pickle'.format(args.data, self.model_name)
            
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_data()
        self.model        = self.add_model()
        self.optimizer    = self.add_optimizer(self.model.parameters())
        
        return 
    
    def _load_data(self, file_path):
        df = pd.read_csv(file_path, sep='\t', header=None, names=None, dtype=str)
        df = df.drop_duplicates()
        return df.values
    
    def load_data(self):
        ''' 
        Load the train, valid and test datasets
        Also load the eval filters
        '''
        data_path = 'data/{0}'.format(self.args.data)
        with open (os.path.join(data_path, 'entities_dict.json'), 'r') as f:
            self.ent_to_id = json.load(f)
        with open (os.path.join(data_path, 'relations_dict.json'), 'r') as f:
            self.rel_to_id = json.load(f)
        self.n_ent = len(list(self.ent_to_id.keys()))
        self.n_rel = len(list(self.rel_to_id.keys()))
        
        self.train_data = self._load_data(os.path.join(data_path, 'train.txt'))
        
        ##### test and valid ####
        self.valid_data = self._load_data(os.path.join(data_path, 'valid.txt'))
        self.test_data = self._load_data(os.path.join(data_path, 'test.txt'))
    
        inp_f = open(os.path.join(data_path, 'to_skip_eval.pickle'), 'rb')
        self.to_skip_eval: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()
        self.to_skip_eval['lhs'] = {(int(k[0]), int(k[1])): v for k,v in self.to_skip_eval['lhs'].items()}
        self.to_skip_eval['rhs'] = {(int(k[0]), int(k[1])): v for k,v in self.to_skip_eval['rhs'].items()}
        #print('To skip eval Lhs: {0}'.format(len(self.to_skip_eval['lhs'])))
        
        return
    
    def add_model(self):
        if self.args.add_reciprocals:
            if self.args.model is None:
                model = Conve(self.args, self.n_ent, 2*self.n_rel)
            elif self.args.model == 'conve':
                model = Conve(self.args, self.n_ent, 2*self.n_rel)
            elif self.args.model == 'distmult':
                model = Distmult(self.args, self.n_ent, 2*self.n_rel)
            elif self.args.model == 'complex':
                model = Complex(self.args, self.n_ent, 2*self.n_rel)
            elif self.args.model == 'transe':
                model = Transe(self.args, self.n_ent, 2*self.n_rel)
            else:
                self.logger.info('Unknown model: {0}', self.args.model)
                raise Exception("Unknown model!")
        else:
            if self.args.model is None:
                model = Conve(self.args, self.n_ent, self.n_rel)
            elif self.args.model == 'conve':
                model = Conve(self.args, self.n_ent, self.n_rel)
            elif self.args.model == 'distmult':
                model = Distmult(self.args, self.n_ent, self.n_rel)
            elif self.args.model == 'complex':
                model = Complex(self.args, self.n_ent, self.n_rel)
            elif self.args.model == 'transe':
                model = Transe(self.args, self.n_ent, self.n_rel)
            else:
                self.logger.info('Unknown model: {0}', self.args.model)
                raise Exception("Unknown model!")
        
        model.to(self.device)
        return model
    
    def add_optimizer(self, parameters):
        #if self.args.optimizer == 'adam' : return torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.lr_decay)
        #else                    : return torch.optim.SGD(parameters,  lr=self.args.lr, weight_decay=self.args.lr_decay)
        return torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.lr_decay)
    
    def save_model(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.args)
        }
        torch.save(state, self.model_path)
        self.logger.info('Saving model to {0}'.format(self.model_path))
        
        return
    
    def load_model(self):
        self.logger.info('Loading saved model from {0}'.format(self.model_path))
        state = torch.load(self.model_path)
        model_params = state['state_dict']
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            self.logger.info(key, size, count)
        
        self.model.load_state_dict(model_params)
        self.optimizer.load_state_dict(state['optimizer'])
        
        return
    
    def evaluate(self, split, batch_size, epoch):
        # run the evaluation - 'split.txt' will be loaded and used for ranking
        self.model.eval()
        with torch.no_grad():
            if self.args.add_reciprocals:
                num_rel= self.n_rel
            else:
                num_rel = 0
            
            if split == 'test':
                results = evaluation(self.model, self.test_data, self.to_skip_eval, 
                                     self.eval_name, num_rel, split, batch_size, epoch, self.device)
            elif split == 'valid':
                results = evaluation(self.model, self.valid_data, self.to_skip_eval, 
                                     self.eval_name, num_rel, split, batch_size, epoch, self.device)
            else:
                data_path = 'data/{0}'.format(self.args.data)
                inp_f = open(os.path.join(data_path, split+'.pickle'), 'rb')
                split_data = np.array(pickle.load(inp_f))
                inp_f.close()
                results = evaluation(self.model, split_data, self.to_skip_eval, 
                                     self.eval_name, num_rel, split, batch_size, epoch, self.device)
            
            
            self.logger.info('[Epoch {} {}]: MRR: lhs : {:.5}, rhs : {:.5}, Avg : {:.5}'.format(epoch, split, results['mrr_lhs'], results['mrr_rhs'], np.mean([results['mrr_lhs'], results['mrr_rhs']])))
        # evaluation has its own logging; so no need to log here
        return results
    
    
    def lp_regularizer(self):
        # Apply p-norm regularization; assign weights to each param
        weight = self.args.reg_weight
        p = self.args.reg_norm
        
        trainable_params = [self.model.emb_e.weight, self.model.emb_rel.weight]
        norm = 0
        for i in range(len(trainable_params)):
            #norm += weight * trainable_params[i].norm(p = p)**p
            norm += weight * torch.sum( torch.abs(trainable_params[i]) ** p)
            
        return norm
        
    def n3_regularizer(self, factors):
        # factors are the embeddings for lhs, rel, rhs for triples in a batch
        weight = self.args.reg_weight
        p = self.args.reg_norm
        
        norm = 0
        for f in factors:
            norm += weight * torch.sum(torch.abs(f) ** p)
            
        return norm / factors[0].shape[0] # scale by number of triples in batch
    
    def get_influence_map(self):
        """
        Turns the influence map into a list, ready to be written to disc. (before: numpy)
        :return: the influence map with lists as values
        """
        assert self.args.save_influence_map == True
        
        for key in self.influence_map:
            self.influence_map[key] = self.influence_map[key].tolist()
        #self.logger.info('get_influence_map passed')
        return self.influence_map
    
    def run_epoch(self, epoch):
        self.model.train()
        losses = []
        
        #shuffle the train dataset
        input_data = torch.from_numpy(self.train_data.astype('int64'))
        actual_examples = input_data[torch.randperm(input_data.shape[0]), :]
        del input_data
        
        batch_size = self.args.train_batch_size
        b_begin = 0
        
        while b_begin < actual_examples.shape[0]:
            self.optimizer.zero_grad()
            input_batch = actual_examples[b_begin: b_begin + batch_size]
            input_batch = input_batch.to(self.device)
            
            s,r,o = input_batch[:,0], input_batch[:,1], input_batch[:,2]
            
            emb_s = self.model.emb_e(s).squeeze(dim=1)
            emb_r = self.model.emb_rel(r).squeeze(dim=1)
            emb_o = self.model.emb_e(o).squeeze(dim=1)
            
            if self.args.add_reciprocals:
                r_rev = r + self.n_rel
                emb_rrev = self.model.emb_rel(r_rev).squeeze(dim=1)
            else:
                r_rev = r
                emb_rrev = emb_r
                
            pred_sr = self.model.forward(emb_s, emb_r, mode='rhs')
            loss_sr = self.model.loss(pred_sr, o) # loss is cross entropy loss
            
            pred_or = self.model.forward(emb_o, emb_rrev, mode='lhs')
            loss_or = self.model.loss(pred_or, s)
            
            total_loss = loss_sr + loss_or
            
            if (self.args.reg_weight != 0.0 and self.args.reg_norm == 3):
                #self.logger.info('Computing regularizer weight')
                if self.args.model == 'complex':
                    emb_dim = self.args.embedding_dim #int(self.args.embedding_dim/2)
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
                    
                total_loss  += self.n3_regularizer(factors_sr)
                total_loss  += self.n3_regularizer(factors_or)
                
            if (self.args.reg_weight != 0.0 and self.args.reg_norm == 2):
                total_loss += self.lp_regularizer()
            
            if self.args.save_influence_map:  # for gradient rollback
                d_loss_emb_s = autograd.grad(total_loss, emb_s,
                                             retain_graph = True)[0]
                d_loss_emb_r = autograd.grad(total_loss, emb_r, 
                                             retain_graph= True)[0]
                d_loss_emb_o = autograd.grad(total_loss, emb_o,
                                             retain_graph= True)[0]
                
                inf_head = d_loss_emb_s.cpu().detach().numpy() # influence of head = gradient of loss due to head
                inf_tail = d_loss_emb_r.cpu().detach().numpy()
                inf_rel = d_loss_emb_o.cpu().detach().numpy()
                
                #print(inf_head.shape, inf_tail.shape, inf_rel.shape)
                
                # need to save the influence per-triple
                for idx in range(input_batch.shape[0]):
                    head, rel, tail = s[idx], r[idx], o[idx]
                    # write the influences to dictionary
                    key_trip = '{0}_{1}_{2}'.format(head.item(), rel.item(), tail.item())
                    key = '{0}_s'.format(key_trip)
                    self.influence_map[key] += inf_head[idx]
                    #self.logger.info('Written to influence map. Key: {0}, Value shape: {1}'.format(key, inf_head.shape))
                    key = '{0}_r'.format(key_trip)
                    self.influence_map[key] += inf_rel[idx]
                    key = '{0}_o'.format(key_trip)
                    self.influence_map[key] += inf_tail[idx]
                    
                    
            total_loss.backward()
            self.optimizer.step()
            losses.append(total_loss.item())
            
            b_begin += batch_size
            
            if (b_begin%5000 == 0) or (b_begin== (actual_examples.shape[0]-1)):
                self.logger.info('[E:{} | {}]: Train Loss:{:.4}'.format(epoch, b_begin, np.mean(losses)))
                
        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss
    
    def fit(self):
        if self.args.resume:
            self.load_model()
            results = self.evaluate(split=self.args.resume_split, batch_size = self.args.test_batch_size, epoch = -1)
            pprint(results)
            
        else:
            self.model.init()
            
        self.logger.info(self.model)
        
        train_losses = []
        for epoch in range(self.args.epochs):
            train_loss = self.run_epoch(epoch)
            train_losses.append(train_loss)
            
            if epoch%20 == 0:
                results_valid = self.evaluate(split='valid', batch_size = self.args.valid_batch_size, epoch = epoch)
                results_test = self.evaluate(split='test', batch_size = self.args.test_batch_size, epoch = epoch)
                self.save_model() # saving model every 20 epochs 
                
                        
            if epoch == (self.args.epochs - 1):
                results_valid = self.evaluate(split='valid', batch_size = self.args.valid_batch_size, epoch = epoch)
                results_test = self.evaluate(split='test', batch_size = self.args.test_batch_size, epoch = epoch)
                self.save_model()
                # save train losses
                #with open(self.loss_path, "wb") as fp:   #Pickling
                #    pickle.dump(train_losses, fp)
                #with open("test.txt", "rb") as fp:   # Unpickling
                #    b = pickle.load(fp)
                
                if self.args.save_influence_map: #save the influence map
#                     with open(self.influence_path, 'w') as out:
#                         out.write(json.dumps(self.get_influence_map(), indent=4)  + '\n')
                    with open(self.influence_path, "wb") as fp:   #Pickling
                        pickle.dump(self.get_influence_map(), fp)
                    self.logger.info('Finished saving influence map')
        
        return
    
if __name__ == '__main__':
    parser = utils.get_argument_parser()
    
    args = parser.parse_args()
    
    if args.reproduce_results:
        args = utils.set_hyperparams(args)
    
    np.set_printoptions(precision=3)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    model = Main(args)
    model.fit()
    
    
    
    
    
    