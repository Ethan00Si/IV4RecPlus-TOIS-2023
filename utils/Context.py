#coding=utf-8

import os
import datetime

import logging
import torch

import config.const as const_util
import trainer
import recommender

import datetime



class ContextManager(object):

    def __init__(self, flags_obj):

        self.exp_name = flags_obj.name
        self.description = flags_obj.description
        self.workspace = flags_obj.workspace
        self.set_default(flags_obj)


    def set_default(self, flags_obj):

        self.set_workspace()
        self.set_logging(flags_obj)

    def set_workspace(self):

        date_time = '_'+str(datetime.datetime.now().month)\
            +'_'+str(datetime.datetime.now().day)\
            +'_'+str(datetime.datetime.now().hour)
        dir_name = self.exp_name + '_' + date_time
        self.workspace = os.path.join(self.workspace, dir_name)
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)




    def set_logging(self, flags_obj):
        # set log file path
        if not os.path.exists(os.path.join(self.workspace, 'log')):
            os.mkdir(os.path.join(self.workspace, 'log'))
        log_file_name = os.path.join(self.workspace, 'log', self.description+'.log')
        logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO, filename=log_file_name, filemode='w')
            

        logging.info('Configs:')
        for flag, value in flags_obj.__dict__.items():
            logging.info('{}: {}'.format(flag, value))
         

    @staticmethod
    def set_trainer(flags_obj, cm,  dm, nc=None):

        model_list1 = ['NRHUB', 'DIN', 'SRGNN']
        model_list2 = ['IV4Rec_UI_DIN', 'IV4Rec_I_DIN', 'IV4Rec_UI_NRHUB',  'IV4Rec_I_NRHUB', 'IV4Rec_UI_SRGNN',  'IV4Rec_I_SRGNN']
        if flags_obj.model in model_list1:
            return trainer.SequenceTrainer(flags_obj, cm, dm, nc)
        elif flags_obj.model in model_list2:
            return trainer.IV4Sequence_Trainer(flags_obj, cm, dm ,nc)
        else:
            raise NotImplementedError(f'Not implement the model: {flags_obj.model}')

    @staticmethod
    def set_recommender(flags_obj, workspace, dm, new_config):

        rec = getattr(recommender, f'{flags_obj.model}_Recommender') (flags_obj, workspace, dm, new_config)
            
        logging.info('model config:')
        for k,v in rec.model_config.items():
            logging.info('{}: {}'.format(k, v))
        
        return rec

    @staticmethod
    def set_device(flags_obj):

        if not flags_obj.use_gpu:
            return torch.device('cpu')
        else:
            return torch.device('cuda:{}'.format(flags_obj.gpu_id))



class DatasetManager(object):

    def __init__(self, flags_obj):

        self.dataset_name = flags_obj.dataset_name
        self.set_dataset()


    def set_dataset(self):
        if self.dataset_name == "mind":
            self.load_path = const_util.mind_path
            self.train_dataset = const_util.mind_train
            self.val_dataset = const_util.mind_validation
            self.test_dataset = const_util.mind_test
            self.user_his_file = const_util.mind_user_browse_his_file
            self.batch_size = const_util.mind_batch_size

            self.qry_per_user = const_util.mind_user_browse_his_file
            self.qry_emb = const_util.mind_qry_emb
            self.item_emb = const_util.mind_item_emb
            self.qry_padding_idx = const_util.mind_qry_padding_idx
            self.rec_padding_idx = const_util.mind_rec_padding_idx

            self.qry_emb_size = const_util.mind_qry_emb_size

            self.rec_his_step = const_util.mind_rec_his_step
            self.src_his_step = const_util.mind_src_his_step
        else:
            raise NotImplementedError('Not support other datasets!')

        logging.info('dataset: {}'.format(self.dataset_name))

    def show(self):
        print(self.__dict__)




class EarlyStopManager(object):

    def __init__(self, config):

        self.min_lr = config['min_lr']
        self.es_patience = config['es_patience']
        self.count = 0
        self.max_metric = 0

    def step(self, lr, metric):

        if lr > self.min_lr:
            if metric > self.max_metric:
                self.max_metric = metric
            return False
        else:
            if metric > self.max_metric:
                self.max_metric = metric
                self.count = 0
                return False
            else:
                self.count = self.count + 1
                if self.count > self.es_patience:
                    return True
                return False


