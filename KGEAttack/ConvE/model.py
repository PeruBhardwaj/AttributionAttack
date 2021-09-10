import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Distmult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Distmult, self).__init__()
        
        if args.max_norm:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, max_norm=1.0)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim)
        else:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
        
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.CrossEntropyLoss()
        
        return
    
    def init(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)
        return
    
    def score_sr(self, sub, rel, sigmoid = False):
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
            
        #sub_emb = self.inp_drop(sub_emb)
        #rel_emb = self.inp_drop(rel_emb) 
        
        pred = torch.mm(sub_emb*rel_emb, self.emb_e.weight.transpose(1,0))
        if sigmoid:
            pred = torch.sigmoid(pred) 
        return pred
    
    def score_or(self, obj, rel, sigmoid = False):
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        
        #obj_emb = self.inp_drop(obj_emb)
        #rel_emb = self.inp_drop(rel_emb) 
        
        pred = torch.mm(obj_emb*rel_emb, self.emb_e.weight.transpose(1,0))
        if sigmoid:
            pred = torch.sigmoid(pred)
        return pred
    
    
    def forward(self, sub_emb, rel_emb, mode='rhs', sigmoid=False):
        '''
        When mode is 'rhs' we expect (s,r); for 'lhs', we expect (o,r)
        For distmult, computations for both modes are equivalent, so we do not need if-else block
        '''
        sub_emb = self.inp_drop(sub_emb)
        rel_emb = self.inp_drop(rel_emb) 
        
        pred = torch.mm(sub_emb*rel_emb, self.emb_e.weight.transpose(1,0))
            
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - score
        '''
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        
        pred = torch.sum(sub_emb*rel_emb*obj_emb, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_emb(self, emb_s, emb_r, emb_o, sigmoid=False):
        '''
        Inputs - embeddings of subject, relation, object
        Return - score
        '''
        pred = torch.sum(emb_s*emb_r*emb_o, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples_vec(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - a vector score for the triple instead of reducing over the embedding dimension
        '''
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        
        pred = sub_emb*rel_emb*obj_emb
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    

class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        
        if args.max_norm:
            self.emb_e = torch.nn.Embedding(num_entities, 2*args.embedding_dim, max_norm=1.0)
            self.emb_rel = torch.nn.Embedding(num_relations, 2*args.embedding_dim)
        else:
            self.emb_e = torch.nn.Embedding(num_entities, 2*args.embedding_dim, padding_idx=None)
            self.emb_rel = torch.nn.Embedding(num_relations, 2*args.embedding_dim, padding_idx=None)
        
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.CrossEntropyLoss()
        
        return
    
    def init(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)
        return
    
    def score_sr(self, sub, rel, sigmoid = False):
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
            
        s_real, s_img = torch.chunk(rel_emb, 2, dim=-1)
        rel_real, rel_img = torch.chunk(sub_emb, 2, dim=-1)
        emb_e_real, emb_e_img = torch.chunk(self.emb_e.weight, 2, dim=-1)

        #s_real = self.inp_drop(s_real)
        #s_img = self.inp_drop(s_img)
        #rel_real = self.inp_drop(rel_real)
        #rel_img = self.inp_drop(rel_img)

        # complex space bilinear product (equivalent to HolE)
#         realrealreal = torch.mm(s_real*rel_real, emb_e_real.transpose(1,0))
#         realimgimg = torch.mm(s_real*rel_img, emb_e_img.transpose(1,0))
#         imgrealimg = torch.mm(s_img*rel_real, emb_e_img.transpose(1,0))
#         imgimgreal = torch.mm(s_img*rel_img, emb_e_real.transpose(1,0))
#         pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        realo_realreal = s_real*rel_real
        realo_imgimg = s_img*rel_img
        realo = realo_realreal - realo_imgimg
        real = torch.mm(realo, emb_e_real.transpose(1,0))
        
        imgo_realimg = s_real*rel_img
        imgo_imgreal = s_img*rel_real
        imgo = imgo_realimg + imgo_imgreal
        img = torch.mm(imgo, emb_e_img.transpose(1,0))
        
        pred = real + img
            
        if sigmoid:
            pred = torch.sigmoid(pred) 
        return pred
    
    
    def score_or(self, obj, rel, sigmoid = False):
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        
        rel_real, rel_img = torch.chunk(rel_emb, 2, dim=-1)
        o_real, o_img = torch.chunk(obj_emb, 2, dim=-1)
        emb_e_real, emb_e_img = torch.chunk(self.emb_e.weight, 2, dim=-1)

        #rel_real = self.inp_drop(rel_real)
        #rel_img = self.inp_drop(rel_img)
        #o_real = self.inp_drop(o_real)
        #o_img = self.inp_drop(o_img)

        # complex space bilinear product (equivalent to HolE)
#         realrealreal = torch.mm(rel_real*o_real, emb_e_real.transpose(1,0))
#         realimgimg = torch.mm(rel_img*o_img, emb_e_real.transpose(1,0))
#         imgrealimg = torch.mm(rel_real*o_img, emb_e_img.transpose(1,0))
#         imgimgreal = torch.mm(rel_img*o_real, emb_e_img.transpose(1,0))
#         pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        reals_realreal = rel_real*o_real
        reals_imgimg = rel_img*o_img
        reals = reals_realreal + reals_imgimg
        real = torch.mm(reals, emb_e_real.transpose(1,0))
        
        imgs_realimg = rel_real*o_img
        imgs_imgreal = rel_img*o_real
        imgs = imgs_realimg - imgs_imgreal
        img = torch.mm(imgs, emb_e_img.transpose(1,0))
        
        pred = real + img
        
        if sigmoid:
            pred = torch.sigmoid(pred)
        return pred
    
    
    def forward(self, sub_emb, rel_emb, mode='rhs', sigmoid=False):
        '''
        When mode is 'rhs' we expect (s,r); for 'lhs', we expect (o,r)
        
        '''
        if mode == 'lhs':
            rel_real, rel_img = torch.chunk(rel_emb, 2, dim=-1)
            o_real, o_img = torch.chunk(sub_emb, 2, dim=-1)
            emb_e_real, emb_e_img = torch.chunk(self.emb_e.weight, 2, dim=-1)
            
            rel_real = self.inp_drop(rel_real)
            rel_img = self.inp_drop(rel_img)
            o_real = self.inp_drop(o_real)
            o_img = self.inp_drop(o_img)
            
            # complex space bilinear product (equivalent to HolE)
#             realrealreal = torch.mm(rel_real*o_real, emb_e_real.transpose(1,0))
#             realimgimg = torch.mm(rel_img*o_img, emb_e_real.transpose(1,0))
#             imgrealimg = torch.mm(rel_real*o_img, emb_e_img.transpose(1,0))
#             imgimgreal = torch.mm(rel_img*o_real, emb_e_img.transpose(1,0))
#             pred = realrealreal + realimgimg + imgrealimg - imgimgreal
            reals_realreal = rel_real*o_real
            reals_imgimg = rel_img*o_img
            reals = reals_realreal + reals_imgimg
            real = torch.mm(reals, emb_e_real.transpose(1,0))

            imgs_realimg = rel_real*o_img
            imgs_imgreal = rel_img*o_real
            imgs = imgs_realimg - imgs_imgreal
            img = torch.mm(imgs, emb_e_img.transpose(1,0))

            pred = real + img
        
        else:
            s_real, s_img = torch.chunk(rel_emb, 2, dim=-1)
            rel_real, rel_img = torch.chunk(sub_emb, 2, dim=-1)
            emb_e_real, emb_e_img = torch.chunk(self.emb_e.weight, 2, dim=-1)
            
            s_real = self.inp_drop(s_real)
            s_img = self.inp_drop(s_img)
            rel_real = self.inp_drop(rel_real)
            rel_img = self.inp_drop(rel_img)
            
            # complex space bilinear product (equivalent to HolE)
#             realrealreal = torch.mm(s_real*rel_real, emb_e_real.transpose(1,0))
#             realimgimg = torch.mm(s_real*rel_img, emb_e_img.transpose(1,0))
#             imgrealimg = torch.mm(s_img*rel_real, emb_e_img.transpose(1,0))
#             imgimgreal = torch.mm(s_img*rel_img, emb_e_real.transpose(1,0))
#             pred = realrealreal + realimgimg + imgrealimg - imgimgreal
            
            realo_realreal = s_real*rel_real
            realo_imgimg = s_img*rel_img
            realo = realo_realreal - realo_imgimg
            real = torch.mm(realo, emb_e_real.transpose(1,0))

            imgo_realimg = s_real*rel_img
            imgo_imgreal = s_img*rel_real
            imgo = imgo_realimg + imgo_imgreal
            img = torch.mm(imgo, emb_e_img.transpose(1,0))

            pred = real + img
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - score
        '''
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        
        s_real, s_img = torch.chunk(sub_emb, 2, dim=-1)
        rel_real, rel_img = torch.chunk(rel_emb, 2, dim=-1)
        o_real, o_img = torch.chunk(obj_emb, 2, dim=-1)
        
        realrealreal = torch.sum(s_real*rel_real*o_real, dim=-1)
        realimgimg = torch.sum(s_real*rel_img*o_img, axis=-1)
        imgrealimg = torch.sum(s_img*rel_real*o_img, axis=-1)
        imgimgreal = torch.sum(s_img*rel_img*o_real, axis=-1)
        
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_emb(self, emb_s, emb_r, emb_o, sigmoid=False):
        '''
        Inputs - embeddings of subject, relation, object
        Return - score
        '''
        
        s_real, s_img = torch.chunk(emb_s, 2, dim=-1)
        rel_real, rel_img = torch.chunk(emb_r, 2, dim=-1)
        o_real, o_img = torch.chunk(emb_o, 2, dim=-1)
        
        realrealreal = torch.sum(s_real*rel_real*o_real, dim=-1)
        realimgimg = torch.sum(s_real*rel_img*o_img, axis=-1)
        imgrealimg = torch.sum(s_img*rel_real*o_img, axis=-1)
        imgimgreal = torch.sum(s_img*rel_img*o_real, axis=-1)
        
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples_vec(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - a vector score for the triple instead of reducing over the embedding dimension
        '''
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        
        s_real, s_img = torch.chunk(sub_emb, 2, dim=-1)
        rel_real, rel_img = torch.chunk(rel_emb, 2, dim=-1)
        o_real, o_img = torch.chunk(obj_emb, 2, dim=-1)
        
        realrealreal = s_real*rel_real*o_real
        realimgimg = s_real*rel_img*o_img
        imgrealimg = s_img*rel_real*o_img
        imgimgreal = s_img*rel_img*o_real
        
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
class Transe(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Transe, self).__init__()
        self.margin = args.transe_margin  #default value is 0.0
        self.norm = args.transe_norm  #default value is 2
        self.args = args
        
        if args.max_norm:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, max_norm=1.0)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim)
        else:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
        
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.CrossEntropyLoss()
        
        return
    
    def init(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)
        return
    
    def score_sr(self, sub, rel, sigmoid = False):
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
            
        #sub_emb = self.inp_drop(sub_emb)
        #rel_emb = self.inp_drop(rel_emb) 
        
        sub_emb = sub_emb.unsqueeze(dim=1)
        rel_emb = rel_emb.unsqueeze(dim=1)
        obj_emb = self.emb_e.weight.unsqueeze(dim=0)
        pred = (sub_emb + rel_emb) - obj_emb
        pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred) 
        return pred
    
    def score_or(self, obj, rel, sigmoid = False):
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        
        #obj_emb = self.inp_drop(obj_emb)
        #rel_emb = self.inp_drop(rel_emb) 
        
        obj_emb = obj_emb.unsqueeze(dim=1) 
        rel_emb = rel_emb.unsqueeze(dim=1)
        sub_emb = self.emb_e.weight.unsqueeze(dim=0)
        pred = sub_emb + (rel_emb - obj_emb)
        pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
            
        if sigmoid:
            pred = torch.sigmoid(pred)
        return pred
    
    
    def forward(self, s_emb, rel_emb, mode='rhs', sigmoid=False):
        '''
        When mode is 'rhs' we expect (s,r); for 'lhs', we expect (o,r)
        '''
        batch_size, num_entities = s_emb.shape[0], self.emb_e.weight.shape[0]
        
        s_emb = self.inp_drop(s_emb)
        rel_emb = self.inp_drop(rel_emb)
        # sub_emb and rel_emb are of shape (num_batches, 1, emb_dim)
        
        if mode == 'lhs':
            # below will be of shape (1, num_entities, emb_dim) to enable broadcast
            #sub_emb = self.emb_e.weight[None,:,:]
            
            #pred = self.emb_e.weight[None,:,:] + (rel_emb - obj_emb)[:,None, :]
            obj_emb = s_emb.unsqueeze(dim=1)
            rel_emb = rel_emb.unsqueeze(dim=1)
            sub_emb = self.emb_e.weight.unsqueeze(dim=0)
            pred = sub_emb + (rel_emb - obj_emb)
            pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
            
        else:
            # below will be of shape (1, num_entities, emb_dim) to enable broadcast
            #obj_emb = self.emb_e.weight[None,:,:]
            
            #pred = (sub_emb + rel_emb)[:,None, :] - self.emb_e.weight[None,:,:]
            sub_emb = s_emb.unsqueeze(dim=1)
            rel_emb = rel_emb.unsqueeze(dim=1)
            obj_emb = self.emb_e.weight.unsqueeze(dim=0)
            pred = (sub_emb + rel_emb) - obj_emb
            pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
            
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - score
        '''
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        
        pred = sub_emb + rel_emb - obj_emb
        pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_emb(self, emb_s, emb_r, emb_o, sigmoid=False):
        '''
        Inputs - embeddings of subject, relation, object
        Return - score
        '''
        pred = emb_s + emb_r - emb_o
        pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
        
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples_vec(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - vector score for the triple instead of reducing over the embedding dimension
        '''
        sub_emb = self.emb_e(sub).squeeze(dim=1)
        rel_emb = self.emb_rel(rel).squeeze(dim=1)
        obj_emb = self.emb_e(obj).squeeze(dim=1)
        
        pred = -(sub_emb + rel_emb - obj_emb)
        pred += torch.tensor(self.margin).to(self.args.device).expand_as(pred)
        #pred = self.margin - torch.norm(pred, p=self.norm, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
        
class Conve(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Conve, self).__init__()
        
        if args.max_norm:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, max_norm=1.0)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim)
        else:
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=None)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=None)
        
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_drop = torch.nn.Dropout2d(args.feat_drop)
        
        self.embedding_dim = args.embedding_dim #default is 200
        self.num_filters = args.num_filters # default is 32
        self.kernel_size = args.kernel_size # default is 3
        self.stack_width = args.stack_width # default is 20
        self.stack_height = args.embedding_dim // self.stack_width
        
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.num_filters)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)

        self.conv1 = torch.nn.Conv2d(1, out_channels=self.num_filters, 
                                     kernel_size=(self.kernel_size, self.kernel_size), 
                                     stride=1, padding=0, bias=args.use_bias)
        #self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias) # <-- default
        
        flat_sz_h = int(2*self.stack_width) - self.kernel_size + 1
        flat_sz_w = self.stack_height - self.kernel_size + 1
        self.flat_sz  = flat_sz_h*flat_sz_w*self.num_filters
        self.fc = torch.nn.Linear(self.flat_sz, args.embedding_dim)
        
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        
        
        self.loss = torch.nn.CrossEntropyLoss()
        
        return
    
    def init(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)
        return
    
    def concat(self, e1_embed, rel_embed, form='plain'):
        if form == 'plain':
            e1_embed = e1_embed. view(-1, 1, self.stack_width, self.stack_height)
            rel_embed = rel_embed.view(-1, 1, self.stack_width, self.stack_height)
            stack_inp = torch.cat([e1_embed, rel_embed], 2)

        elif form == 'alternate':
            e1_embed = e1_embed. view(-1, 1, self.embedding_dim)
            rel_embed = rel_embed.view(-1, 1, self.embedding_dim)
            stack_inp = torch.cat([e1_embed, rel_embed], 1)
            stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.stack_width, self.stack_height))

        else: raise NotImplementedError
        return stack_inp
    
    def conve_architecture(self, sub_emb, rel_emb):
        stacked_inputs = self.concat(sub_emb, rel_emb)
        stacked_inputs = self.bn0(stacked_inputs)
        x  = self.inp_drop(stacked_inputs)
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = F.relu(x)
        x  = self.feature_drop(x)
        #x  = x.view(x.shape[0], -1)
        x  = x.view(-1, self.flat_sz)
        x  = self.fc(x)
        x  = self.hidden_drop(x)
        x  = self.bn2(x)
        x  = F.relu(x)
        
        return x
    
    def score_sr(self, sub, rel, sigmoid = False):
        sub_emb = self.emb_e(sub)
        rel_emb = self.emb_rel(rel)
        
        x = self.conve_architecture(sub_emb, rel_emb)
        
        pred = torch.mm(x, self.emb_e.weight.transpose(1,0))
        pred += self.b.expand_as(pred) 
        
        if sigmoid:
            pred = torch.sigmoid(pred) 
        return pred
    
    def score_or(self, obj, rel, sigmoid = False):
        obj_emb = self.emb_e(obj)
        rel_emb = self.emb_rel(rel)
        
        x = self.conve_architecture(obj_emb, rel_emb)
        pred = torch.mm(x, self.emb_e.weight.transpose(1,0))
        pred += self.b.expand_as(pred)
        
        if sigmoid:
            pred = torch.sigmoid(pred)
        return pred
    
    
    def forward(self, sub_emb, rel_emb, mode='rhs', sigmoid=False):
        '''
        When mode is 'rhs' we expect (s,r); for 'lhs', we expect (o,r)
        For conve, computations for both modes are equivalent, so we do not need if-else block
        '''
        x = self.conve_architecture(sub_emb, rel_emb)
        
        pred = torch.mm(x, self.emb_e.weight.transpose(1,0))
        pred += self.b.expand_as(pred)
            
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - score
        '''
        sub_emb = self.emb_e(sub)
        rel_emb = self.emb_rel(rel)
        obj_emb = self.emb_e(obj)
        x = self.conve_architecture(sub_emb, rel_emb)
        
        pred = torch.mm(x, obj_emb.transpose(1,0))
        #print(pred.shape)
        pred += self.b[obj].expand_as(pred) #taking the bias value for object embedding
        # above works fine for single input triples; 
        # but if input is batch of triples, then this is a matrix of (num_trip x num_trip) where diagonal is scores
        # so use torch.diagonal() after calling this function
        pred = torch.diagonal(pred)
        # or could have used : pred= torch.sum(x*obj_emb, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_emb(self, emb_s, emb_r, emb_o, sigmoid=False):
        '''
        Inputs - embeddings of subject, relation, object
        Return - score
        '''
        x = self.conve_architecture(emb_s, emb_r)
        
        pred = torch.mm(x, emb_o.transpose(1,0))
        #pred += self.b[obj].expand_as(pred) #taking the bias value for object embedding - don't know which obj
        # above works fine for single input triples; 
        # but if input is batch of triples, then this is a matrix of (num_trip x num_trip) where diagonal is scores
        # so use torch.diagonal() after calling this function
        pred = torch.diagonal(pred)
        # or could have used : pred= torch.sum(x*obj_emb, dim=-1)
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
    def score_triples_vec(self, sub, rel, obj, sigmoid=False):
        '''
        Inputs - subject, relation, object
        Return - a vector score for the triple instead of reducing over the embedding dimension
        '''
        sub_emb = self.emb_e(sub)
        rel_emb = self.emb_rel(rel)
        obj_emb = self.emb_e(obj)
        
        x = self.conve_architecture(sub_emb, rel_emb)
        
        #pred = torch.mm(x, obj_emb.transpose(1,0))
        pred = x*obj_emb
        #print(pred.shape, self.b[obj].shape) #shapes are [7,200] and [7]
        #pred += self.b[obj].expand_as(pred) #taking the bias value for object embedding - can't add scalar to vector
        
        #pred = sub_emb*rel_emb*obj_emb
        
        if sigmoid:
            pred = torch.sigmoid(pred)

        return pred
    
        
        