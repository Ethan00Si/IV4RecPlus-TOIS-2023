import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from .data_sampler import *
from .data_utils import *

class BaseDataset(Dataset):
    
    def __init__(self):
        super(Dataset, self).__init__()

    def make_sampler(self, flags_obj):

        raise NotImplementedError

    def __len__(self):

        return self.sampler.record.shape[0]

    def __getitem__(self, index):

        raise NotImplementedError


class PointDataset(BaseDataset):
    def __init__(self, flags_obj) :
        '''
        data format: 
            uID	 itemID	 label	unixTime
        '''
        super(BaseDataset, self).__init__()
        self.make_sampler(flags_obj)

    def make_sampler(self, flags_obj):
        self.sampler = PointSampler(flags_obj)
    
    def __getitem__(self, index):
        
        users, items, labels = self.sampler.sample(index)

        return users, items, torch.tensor(labels, dtype=torch.float32)




class Sequence_Dataset(BaseDataset):
    def __init__(self, flags_obj):
        'for MIND dataset. invariable history length'
        super().__init__()
        self.make_sampler(flags_obj)

    def make_sampler(self, flags_obj):
        self.sampler = Sequence_Sampler(flags_obj)

    def __getitem__(self, index):
        users, items, labels, rec_his, src_his, item_qry = self.sampler.sample(index)

        return users, items, rec_his, src_his, item_qry, torch.tensor(labels,dtype=torch.float32)

    @staticmethod
    def get__emb( load_path, qry_emb_file, item_emb_file):
        npy_loader = NpyLoader(load_path)
        qry_emb = npy_loader.load(qry_emb_file, allow_pickle=True)
        item_emb = npy_loader.load(item_emb_file, allow_pickle=True)
        return qry_emb, item_emb


class Test_Dataset(Dataset):
    def __init__(self, config):
        '''for MIND dataset'''
        super().__init__()
        tsv_loader = TsvLoader(config['load_path'])
        # user item interactions
        self.record = tsv_loader.load(filename=config['dataset_file'], sep='\t')
        self.record = self.record.values

        self.is_sequence = config['is_sequence']

        if self.is_sequence:

            json_loader = JsonLoader(config['load_path'])
            # user search history
            self.user_src_his = json_loader.load(config['user_src_his_file'])
            self.user_src_his = {int(k):v for k,v in self.user_src_his.items()}

            self.user_rec_his = json_loader.load(config['user_rec_his_file'])
            self.user_rec_his = {int(k):v for k,v in self.user_rec_his.items()}


       
    def __len__(self):
        return self.record.shape[0]

    def __getitem__(self, index):

        user, item, label, impression_ID = self.record[index]
        user, item, impression_ID = int(user), int(item), int(impression_ID)

        if self.is_sequence == False:
            return user, item, float(label)
        else:
            #history
            rec_his = self.user_rec_his[user]
            src_his = rec_his
            item_qry = item

            return impression_ID, user, item, np.array(rec_his), np.array(src_his), item_qry,  float(label)




class SRGNN_Sequence_Dataset(Sequence_Dataset): # todo
    def __init__(self, flags_obj, item_padding_idx):
        'for Kwai dataset. invariable history length'
        super().__init__(flags_obj)
        self.item_padding_idx = item_padding_idx

    def make_sampler(self, flags_obj):
        self.sampler = SRGNN_Sequence_Sampler(flags_obj)

    def __getitem__(self, index):
        users, item, labels, rec_his, src_his, item_qry, alias_inputs, A, items, mask = self.sampler.sample(index)
        return users, item, rec_his, src_his, item_qry, torch.tensor(labels, dtype=torch.float32), alias_inputs, A, items, mask



class SRGNN_Test_Dataset(object): # todo
    def __init__(self, config, item_padding_idx):
        '''for MIND dataset'''
        super(SRGNN_Test_Dataset, self).__init__()
        self.item_padding_idx = item_padding_idx
        tsv_loader = TsvLoader(config['load_path'])
        # user item interactions
        self.record = tsv_loader.load(filename=config['dataset_file'], sep='\t')
        self.record = self.record.values

        self.is_sequence = config['is_sequence']

        if self.is_sequence:
            config['user_src_his_file'] = config['user_src_his_file'][:-5]+'_SRGNN.npy'
            config['user_rec_his_file'] = config['user_rec_his_file'][:-5]+'_SRGNN.npy'
            # user search history
            # self.user_src_his = np.load(config['load_path']+config['user_src_his_file'], allow_pickle=True)
            # self.user_src_his = {int(k): v for k, v in self.user_src_his.item().items()}

            self.user_rec_his = np.load(config['load_path']+'/'+config['user_rec_his_file'], allow_pickle=True)
            self.user_rec_his = {int(k): v for k, v in self.user_rec_his.item().items()}

    def __len__(self):
        return self.record.shape[0]
    def __getitem__(self, index):

        user, item, label, impression_ID = self.record[index]
        user, item, impression_ID = int(user), int(item), int(impression_ID)

        if self.is_sequence == False:
            return user, item, float(label)
        else:
            # history
            his_data = self.user_rec_his[user]
            rec_his, alias_inputs, A, items, mask = his_data[0], his_data[1], his_data[2], his_data[3], his_data[4]
            rec_his, alias_inputs, A, items, mask = np.array(rec_his, dtype=np.int64), np.array(alias_inputs), A.A, np.array(items, dtype=np.int64), np.array(mask)
            src_his = rec_his
            item_qry = item

            return impression_ID, user, item, np.array(rec_his), np.array(src_his), item_qry, alias_inputs, A, items, mask, float(label)




GLOBAL_SEED = 1
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def get_dataloader(data_set, bs, **kwargs):
    return DataLoader( data_set, batch_size = bs, prefetch_factor = 3,
    shuffle=False, pin_memory = False, num_workers = 16, worker_init_fn=worker_init_fn, **kwargs
    )
