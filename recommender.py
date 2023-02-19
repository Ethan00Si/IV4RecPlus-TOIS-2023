#coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import utils.data as data
from utils import Context as ctxt
import config.const as const_util
from models import *

import os
import yaml
import numpy as np
import torchsnooper

class Recommender(object):

    def __init__(self, flags_obj, workspace, dm, nc=None):

        self.dm = dm # dataset manager
        self.model_name = flags_obj.model
        self.flags_obj = flags_obj
        self.set_device()
        self.load_model_config()
        self.update_model_config(nc)
        self.set_model()
        self.workspace = workspace

    def set_device(self):

        self.device  = ctxt.ContextManager.set_device(self.flags_obj)

    def load_model_config(self):
        path = 'config/{}_{}.yaml'.format(self.model_name, self.dm.dataset_name)
        f = open(path)
        self.model_config = yaml.load(f, Loader=yaml.FullLoader)


    def update_model_config(self, new_config):
        if new_config is not None:
            for key in [item for item in new_config.keys() if item in self.model_config.keys()]:
                if type(self.model_config[key]) == dict:
                    self.model_config[key].update(new_config[key])
                else:
                    self.model_config[key] = new_config[key]

    def set_model(self):

        raise NotImplementedError

    def transfer_model(self):

        self.model = self.model.to(self.device)

    def save_ckpt(self):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'best.pth')
        torch.save(self.model.state_dict(), model_path)

    def load_ckpt(self, assigned_path=None):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        model_path = None
        if assigned_path is not None:
            model_path = assigned_path
        else:   
            model_path = os.path.join(ckpt_path, 'best.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def get_dataloader(self):

        raise NotImplementedError

    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'])

    def predict(self, sample):
        '''generate prediction'''

        raise NotImplementedError

    def get_loss(self, sample):

        raise NotImplementedError

    def get_sequence_dataloader(self):
        
        
        if self.dm.dataset_name == 'mind':
            mind_train_dataset_setting = {'dataset_file':self.dm.train_dataset,
            'user_his_file':self.dm.user_his_file, 
            'load_path':self.dm.load_path,
            'user_src_his_file':self.dm.qry_per_user
            }
            dst = data.Sequence_Dataset(mind_train_dataset_setting)
        
        dld = data.get_dataloader(dst, bs = self.dm.batch_size)

        return dld



class NRHUB_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)

    def get_dataloader(self):
        
        return self.get_sequence_dataloader()

    def set_model(self):

        self.dataloader = self.get_dataloader()
        
        if self.dm.dataset_name == 'mind':
            qry_emb, item_emb = data.Sequence_Dataset.get__emb(self.dm.load_path, self.dm.qry_emb, self.dm.item_emb)
            qry_emb = torch.tensor(qry_emb, dtype=torch.float32)
            item_emb = torch.tensor(item_emb, dtype=torch.float32)

            self.model = NRHUB(item_emb, qry_emb, rec_pad=self.dm.rec_padding_idx,
                src_pad=self.dm.qry_padding_idx, qry_emb_size=self.dm.qry_emb_size, config=self.model_config
            )

        self.loss_func = nn.BCELoss()

    # @torchsnooper.snoop()
    def predict(self, sample):
    
        user, item, rec_his, src_his, item_qry = sample

        # user = user.to(self.device)
        item = item.to(self.device)
        rec_his = rec_his.to(self.device)
        src_his = src_his.to(self.device)

        score = self.model.forward(item, rec_his, src_his)
        
        return score
        
    def get_loss(self, sample):
    
        data, label = sample[:-1], sample[-1]
        pred = self.predict(data)
        label = label.to(pred.device)

        loss = self.loss_func(pred, label)

        return loss

class IV4Rec_UI_NRHUB_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)

    def get_dataloader(self):
        return self.get_sequence_dataloader()

    def set_model(self):
        self.dataloader = self.get_dataloader()

        if self.dm.dataset_name == 'mind' :
            qry_emb, item_emb = data.Sequence_Dataset.get__emb(self.dm.load_path, self.dm.qry_emb, self.dm.item_emb)
            item_emb = torch.tensor(item_emb, dtype=torch.float32)
            qry_emb = torch.tensor(qry_emb, dtype=torch.float32)

            self.model = IV4Rec_UI_NRHUB(item_emb, self.dm.rec_padding_idx,
                                   config=self.model_config, qry_emb_matrix=qry_emb, qry_padding_idx=self.dm.qry_padding_idx)

        self.loss_func = nn.BCELoss()

    def get_loss(self, sample):
        user, item, rec_his, src_his, item_qry, label = sample
        item, rec_his, src_his, item_qry, label = item.to(self.device), rec_his.to(self.device), src_his.to(
            self.device), item_qry.to(self.device), label.to(self.device)
        loss = self.model(rec_his, src_his, item, item_qry, label)

        return loss

    # @torchsnooper.snoop()
    def predict(self, sample):
        user, item, rec_his, src_his, item_qry = sample
        item, rec_his, src_his, item_qry = item.to(self.device), rec_his.to(self.device), src_his.to(
            self.device), item_qry.to(self.device)

        score = self.model.predict(rec_his, src_his, item, item_qry)

        return score

class IV4Rec_I_NRHUB_Recommender(IV4Rec_UI_NRHUB_Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)
    
    def set_model(self):

        if self.dm.dataset_name == 'mind' :
            self.dataloader = self.get_dataloader()
            qry_emb, item_emb = data.Sequence_Dataset.get__emb(self.dm.load_path, self.dm.qry_emb, self.dm.item_emb)
            item_emb = torch.tensor(item_emb, dtype=torch.float32)
            qry_emb = torch.tensor(qry_emb, dtype=torch.float32)

            self.model = IV4Rec_I_NRHUB(item_emb, self.dm.rec_padding_idx,
                                   config=self.model_config, qry_emb_matrix=qry_emb, qry_padding_idx=self.dm.qry_padding_idx)

        else:
            raise ValueError('IV4Rec_UI_NRHUB only for MIND dataset')

        self.loss_func = nn.BCELoss()

    def get_loss(self, sample):
        user, item, rec_his, src_his, item_qry, label = sample
        item, rec_his, src_his, item_qry, label = item.to(self.device), rec_his.to(self.device), src_his.to(
            self.device), item_qry.to(self.device), label.to(self.device)
        loss = self.model(rec_his, src_his, item, item_qry, label, src_his)

        return loss

    # @torchsnooper.snoop()
    def predict(self, sample):
        user, item, rec_his, src_his, item_qry = sample
        item, rec_his, src_his, item_qry = item.to(self.device), rec_his.to(self.device), src_his.to(
            self.device), item_qry.to(self.device)

        score = self.model.predict(rec_his, src_his, item, item_qry, src_his)

        return score

class DIN_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)

    def get_dataloader(self):
        
        return self.get_sequence_dataloader()

    def set_model(self):
    
        self.dataloader = self.get_dataloader()
        
        if self.dm.dataset_name == 'mind':
            _, item_emb = data.Sequence_Dataset.get__emb(self.dm.load_path, self.dm.qry_emb, self.dm.item_emb)
            item_emb = torch.tensor(item_emb, dtype=torch.float32)

            self.model = DeepInterestNetwork(item_emb, self.dm.rec_padding_idx, config=self.model_config)

        self.loss_func = nn.BCELoss()

        
    def get_loss(self, sample):
    
        user, item, rec_his, src_his, item_qry, label = sample
        item, rec_his, label = item.to(self.device), rec_his.to(self.device), label.to(self.device)
        loss = self.model(rec_his, item, label)

        return loss

    # @torchsnooper.snoop()
    def predict(self, sample):
    
        user, item, rec_his, src_his, item_qry = sample
        # user = user.to(self.device)
        item = item.to(self.device)
        rec_his = rec_his.to(self.device)
        # src_his = src_his.to(self.device)

        score = self.model.predict(rec_his, item)
        
        return score


class IV4Rec_UI_DIN_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)

    def get_dataloader(self):
        
        return self.get_sequence_dataloader()

    def set_model(self):
        
        self.dataloader = self.get_dataloader()
        
        if self.dm.dataset_name == 'mind':
            qry_emb, item_emb = data.Sequence_Dataset.get__emb(self.dm.load_path, self.dm.qry_emb, self.dm.item_emb)
            item_emb = torch.tensor(item_emb, dtype=torch.float32)
            qry_emb = torch.tensor(qry_emb, dtype=torch.float32)

            self.model = IV4Rec_UI_DIN(item_emb_matrix=item_emb, rec_padding_idx=self.dm.rec_padding_idx, 
                config = self.model_config, qry_emb_matrix=qry_emb, qry_padding_idx=self.dm.qry_padding_idx)


        self.loss_func = nn.BCELoss()

    def get_loss(self, sample):
        
        user, item, rec_his, src_his, item_qry, label = sample
        item, rec_his, src_his, item_qry, label = item.to(self.device), rec_his.to(self.device), src_his.to(self.device), item_qry.to(self.device), label.to(self.device)
        loss = self.model(rec_his, src_his, item, item_qry, label)

        return loss

    # @torchsnooper.snoop()
    def predict(self, sample):
    
        user, item, rec_his, src_his, item_qry = sample
        # user = user.to(self.device)
        item = item.to(self.device)
        rec_his = rec_his.to(self.device)
        src_his = src_his.to(self.device)
        item_qry = item_qry.to(self.device)

        score = self.model.predict(rec_his, src_his, item, item_qry)
        
        return score

class IV4Rec_I_DIN_Recommender(IV4Rec_UI_DIN_Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)

    def set_model(self):
        
        self.dataloader = self.get_dataloader()
        
        if self.dm.dataset_name == 'mind':
            qry_emb, item_emb = data.Sequence_Dataset.get__emb(self.dm.load_path, self.dm.qry_emb, self.dm.item_emb)
            item_emb = torch.tensor(item_emb, dtype=torch.float32)
            qry_emb = torch.tensor(qry_emb, dtype=torch.float32)

            self.model = IV4Rec_I_DIN(item_emb_matrix=item_emb, rec_padding_idx=self.dm.rec_padding_idx, 
                config = self.model_config, qry_emb_matrix=qry_emb, qry_padding_idx=self.dm.qry_padding_idx)


        self.loss_func = nn.BCELoss()

class SRGNN_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)

    def get_sequence_dataloader(self):

        if self.dm.dataset_name == 'mind':
            mind_train_dataset_setting = {'dataset_file': self.dm.train_dataset,
                                          'user_his_file': self.dm.user_his_file[:-5]+'_SRGNN.npy',
                                          'load_path': self.dm.load_path,
                                          'user_src_his_file': self.dm.qry_per_user[:-5]+'_SRGNN.npy'
                                          }
            dst = data.SRGNN_Sequence_Dataset(mind_train_dataset_setting, self.dm.rec_padding_idx)

        dld = data.get_dataloader(dst, bs=self.dm.batch_size)

        return dld

    def get_dataloader(self):
        return self.get_sequence_dataloader()

    def set_model(self):
        self.dataloader = self.get_dataloader()

        if self.dm.dataset_name == 'mind':
            qry_emb, item_emb = data.SRGNN_Sequence_Dataset.get__emb(self.dm.load_path, self.dm.qry_emb, self.dm.item_emb)
        item_emb = torch.tensor(item_emb, dtype=torch.float32)
        qry_emb = torch.tensor(qry_emb, dtype=torch.float32)

        self.model = SRGNN(item_emb, self.dm.rec_padding_idx, rec_padding_len=self.dm.rec_his_step,
                           config=self.model_config, qry_emb=qry_emb, qry_padding_idx=self.dm.qry_padding_idx)

        self.loss_func = nn.BCELoss()

    def get_loss(self, sample):
        user, item, rec_his, src_his, item_qry, label, alias_inputs, A, items, mask = sample
        item, rec_his, src_his, item_qry, label, alias_inputs, A, items, mask = item.to(self.device), rec_his.to(self.device), src_his.to(
            self.device), item_qry.to(self.device), label.to(self.device), alias_inputs.to(self.device), A.to(self.device).to(torch.float32), items.to(self.device), mask.to(self.device)

        loss = self.model(rec_his, src_his, item, item_qry, label, alias_inputs, A, items, mask)

        return loss

    # @torchsnooper.snoop()
    def predict(self, sample):
        user, item, rec_his, src_his, item_qry,  alias_inputs, A, items, mask = sample
        # user = user.to(self.device)
        item = item.to(self.device)
        rec_his = rec_his.to(self.device)
        src_his = src_his.to(self.device)
        item_qry = item_qry.to(self.device)
        alias_inputs = alias_inputs.to(self.device)
        A = A.to(self.device).to(torch.float32)
        items = items.to(self.device).to(torch.int)
        mask = mask.to(self.device)

        score = self.model.predict(rec_his, src_his, item, item_qry, alias_inputs, A, items, mask)

        return score


class IV4Rec_UI_SRGNN_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)

    def get_sequence_dataloader(self):

        if self.dm.dataset_name == 'mind':
            mind_train_dataset_setting = {'dataset_file': self.dm.train_dataset,
                                          'user_his_file': self.dm.user_his_file[:-5]+'_SRGNN.npy',
                                          'load_path': self.dm.load_path,
                                          'user_src_his_file': self.dm.qry_per_user[:-5]+'_SRGNN.npy'
                                          }
            dst = data.SRGNN_Sequence_Dataset(mind_train_dataset_setting, self.dm.rec_padding_idx)

        dld = data.get_dataloader(dst, bs=self.dm.batch_size)

        return dld

    def get_dataloader(self):
        return self.get_sequence_dataloader()

    def set_model(self):
        self.dataloader = self.get_dataloader()

        if self.dm.dataset_name == 'mind':
            qry_emb, item_emb = data.SRGNN_Sequence_Dataset.get__emb(self.dm.load_path, self.dm.qry_emb, self.dm.item_emb)
        item_emb = torch.tensor(item_emb, dtype=torch.float32)
        qry_emb = torch.tensor(qry_emb, dtype=torch.float32)
        self.model = IV4Rec_UI_SRGNN(item_emb, self.dm.rec_padding_idx, rec_padding_len=self.dm.rec_his_step,
                           config=self.model_config, qry_emb=qry_emb, qry_padding_idx=self.dm.qry_padding_idx)

        self.loss_func = nn.BCELoss()

    def get_loss(self, sample):
        user, item, rec_his, src_his, item_qry, label, alias_inputs, A, items, mask = sample
        item, rec_his, src_his, item_qry, label, alias_inputs, A, items, mask = item.to(self.device), rec_his.to(self.device), src_his.to(
            self.device), item_qry.to(self.device), label.to(self.device), alias_inputs.to(self.device), A.to(self.device).to(torch.float32), items.to(self.device), mask.to(self.device)

        loss = self.model(rec_his, src_his, item, item_qry, label, alias_inputs, A, items, mask)

        return loss

    # @torchsnooper.snoop()
    def predict(self, sample):
        user, item, rec_his, src_his, item_qry,  alias_inputs, A, items, mask = sample
        # user = user.to(self.device)
        item = item.to(self.device)
        rec_his = rec_his.to(self.device)
        src_his = src_his.to(self.device)
        item_qry = item_qry.to(self.device)
        alias_inputs = alias_inputs.to(self.device)
        A = A.to(self.device).to(torch.float32)
        items = items.to(self.device).to(torch.int)
        mask = mask.to(self.device)

        score = self.model.predict(rec_his, src_his, item, item_qry, alias_inputs, A, items, mask)

        return score

class IV4Rec_I_SRGNN_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)

    def get_sequence_dataloader(self):

        if self.dm.dataset_name == 'mind':
            mind_train_dataset_setting = {'dataset_file': self.dm.train_dataset,
                                          'user_his_file': self.dm.user_his_file[:-5]+'_SRGNN.npy',
                                          'load_path': self.dm.load_path,
                                          'user_src_his_file': self.dm.qry_per_user[:-5]+'_SRGNN.npy'
                                          }
            dst = data.SRGNN_Sequence_Dataset(mind_train_dataset_setting, self.dm.rec_padding_idx)

        dld = data.get_dataloader(dst, bs=self.dm.batch_size)

        return dld

    def get_dataloader(self):
        return self.get_sequence_dataloader()



    def set_model(self):
        self.dataloader = self.get_dataloader()

        if self.dm.dataset_name == 'mind':
            qry_emb, item_emb = data.SRGNN_Sequence_Dataset.get__emb(self.dm.load_path, self.dm.qry_emb, self.dm.item_emb)
        item_emb = torch.tensor(item_emb, dtype=torch.float32)
        qry_emb = torch.tensor(qry_emb, dtype=torch.float32)
        self.model = IV4Rec_I_SRGNN(item_emb, self.dm.rec_padding_idx, rec_padding_len=self.dm.rec_his_step,
                           config=self.model_config, qry_emb=qry_emb, qry_padding_idx=self.dm.qry_padding_idx)

        self.loss_func = nn.BCELoss()

    def get_loss(self, sample):
        user, item, rec_his, src_his, item_qry, label, alias_inputs, A, items, mask = sample
        item, rec_his, src_his, item_qry, label, alias_inputs, A, items, mask = item.to(self.device), rec_his.to(self.device), src_his.to(
            self.device), item_qry.to(self.device), label.to(self.device), alias_inputs.to(self.device), A.to(self.device).to(torch.float32), items.to(self.device), mask.to(self.device)

        loss = self.model(rec_his, src_his, item, item_qry, label, alias_inputs, A, items, mask)

        return loss

    # @torchsnooper.snoop()
    def predict(self, sample):
        user, item, rec_his, src_his, item_qry,  alias_inputs, A, items, mask = sample
        # user = user.to(self.device)
        item = item.to(self.device)
        rec_his = rec_his.to(self.device)
        src_his = src_his.to(self.device)
        item_qry = item_qry.to(self.device)
        alias_inputs = alias_inputs.to(self.device)
        A = A.to(self.device).to(torch.float32)
        items = items.to(self.device).to(torch.int)
        mask = mask.to(self.device)

        score = self.model.predict(rec_his, src_his, item, item_qry, alias_inputs, A, items, mask)

        return score
