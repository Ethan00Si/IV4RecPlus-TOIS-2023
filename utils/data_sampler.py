
import imp
import numpy as np
import pandas as pd
from .data_utils import NpyLoader, TsvLoader, JsonLoader
import ast


class Sampler(object):
    
    def __init__(self, dataset_file, user_his_file, load_path):

        tsv_loader = TsvLoader(load_path)
        json_loader = JsonLoader(load_path)

        # user item interactions
        self.record = tsv_loader.load(filename=dataset_file, sep='\t')

        # user browse history
        self.user_his = json_loader.load(user_his_file)
        self.user_his = {int(k):v for k,v in self.user_his.items()}
    
        self.n_user = self.record.uID.unique().max()+1 
        self.n_item = self.record.itemID.unique().max()+1 

        self.record = self.record.values
    
    def sample(self, index, **kwargs):

        raise NotImplementedError


class PointSampler(Sampler):
    
    def __init__(self, flags_object):

        super(PointSampler, self).__init__(flags_object['dataset_file'],\
            flags_object['user_his_file'], flags_object['load_path'])
    
    def sample(self, index):

        user = int(self.record[index][0])
        item = int(self.record[index][1])
        label = float(self.record[index][2])
        
        return user, item, label


class Sequence_Sampler(Sampler):
    def __init__(self, flags_object):
        '''for MIND dataset. invariable history length'''
        super().__init__(flags_object['dataset_file'],\
            flags_object['user_his_file'], flags_object['load_path'])


    def sample(self, index):

        user = int(self.record[index][0])
        item = int(self.record[index][1])
        label = float(self.record[index][2])
        
        #history
        rec_his = self.user_his[user]
        rec_his = np.array(rec_his, dtype=np.int64)
        
        src_his = rec_his
        item_qry = item
        

        return user, item, label, rec_his, src_his, item_qry



class SRGNN_Sequence_Sampler(object): # todo
    def __init__(self, flags_object):
        '''for MIND dataset. invariable history length'''
        dataset_file, user_his_file, load_path = flags_object['dataset_file'], \
                         flags_object['user_his_file'], flags_object['load_path']
        tsv_loader = TsvLoader(load_path)
        json_loader = JsonLoader(load_path)

        # user item interactions
        self.record = tsv_loader.load(filename=dataset_file, sep='\t')

        # user browse history
        self.user_his = np.load(load_path+'/'+user_his_file, allow_pickle=True)
        self.user_his = {int(k): v for k, v in self.user_his.item().items()}

        self.n_user = self.record.uID.unique().max() + 1  
        self.n_item = self.record.itemID.unique().max() + 1

        self.record = self.record.values
        
    def sample(self, index):
        user = int(self.record[index][0])
        item = int(self.record[index][1])
        label = float(self.record[index][2])

        # history
        his_data = self.user_his[user]

        rec_his, alias_inputs, A, items, mask = his_data[0], his_data[1], his_data[2], his_data[3], his_data[4]
        rec_his, alias_inputs, A, items, mask = np.array(rec_his, dtype=np.int64), np.array(alias_inputs), A.A, np.array(items), np.array(mask)

        src_his = rec_his
        item_qry = item

        return user, item, label, rec_his, src_his, item_qry, alias_inputs, A, items, mask
