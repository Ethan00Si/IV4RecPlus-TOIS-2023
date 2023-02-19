#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from tqdm import tqdm

from utils import Context as ctxt
from tester import Tester

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import logging


class Trainer(object):

    def __init__(self, flags_obj, cm,  dm, new_config=None):

        self.name = flags_obj.name + '_trainer'
        self.cm = cm #context manager
        self.dm = dm #dataset manager
        self.flags_obj = flags_obj
        self.set_recommender(flags_obj, cm.workspace, dm, new_config)
        self.recommender.transfer_model()
        self.lr = self.recommender.model_config['lr']
        self.set_tensorboard(flags_obj.tb)
        self.tester = Tester(flags_obj, self.recommender, self.writer)
        

    def set_recommender(self, flags_obj, workspace, dm, new_config):

        self.recommender = ctxt.ContextManager.set_recommender(flags_obj, workspace, dm, new_config)

    def train(self):

        self.set_dataloader()
        self.set_optimizer()
        self.set_scheduler()
        self.set_esm() #early stop manager

        best_metric = 0
        train_loss = [0.0, 0.0, 0.0, 0.0]
        val_loss = [0.0]

        for epoch in range(self.flags_obj.epochs):

            self.train_one_epoch(epoch, train_loss)
            watch_metric_value = self.validate(epoch, val_loss)
            if watch_metric_value > best_metric:
                self.recommender.save_ckpt()
                logging.info('save ckpt at epoch {}'.format(epoch))
                best_metric = watch_metric_value
            self.scheduler.step(watch_metric_value)

            stop = self.esm.step(self.lr, watch_metric_value)
            if stop:
                break

    def set_test_dataloader(self):
        raise NotImplementedError

    def test(self, assigned_model_path = None, load_config=True):

        self.set_test_dataloader()

        # if self.flags_obj.test_model == 'best':
        if load_config:
            self.recommender.load_ckpt(assigned_path = assigned_model_path)

        total_loss = [0.0]
        results = self.tester.test(total_loss=total_loss, mode='test')

        logging.info('TEST results :')
        self.record_metrics('final', results)
        print('test: ', results)

    def set_dataloader(self):

        raise NotImplementedError

    def set_optimizer(self):

        self.optimizer = self.recommender.get_optimizer()

    def set_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
         mode='max', patience=self.recommender.model_config['patience'], 
         min_lr=self.recommender.model_config['min_lr'])

    def set_esm(self):

        self.esm = ctxt.EarlyStopManager(self.recommender.model_config)

    def set_tensorboard(self, tb=False):
        import os
        if tb:
            self.writer = SummaryWriter("{}/tb/".format(
                os.path.join(self.cm.workspace, 'log')))
        else:
            self.writer = None


    def record_metrics(self, epoch, metric):    

        logging.info('VALIDATION epoch: {}, results: {}'.format(epoch, metric))
        if self.writer:
            if epoch != 'final':
                for k,v in metric.items():
                        self.writer.add_scalar("metric/"+str(k), v, epoch)

    def adapt_hyperparameters(self, epoch):

        raise NotImplementedError

    def train_one_epoch(self, epoch, train_loss):

        self.lr = self.train_one_epoch_core(epoch, self.dataloader, self.optimizer, self.lr, train_loss)

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr, train_loss):

        epoch_loss = train_loss[0]

        self.recommender.model.train()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:

            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        tqdm_ = tqdm(iterable=dataloader, mininterval=1, ncols=100)
        for step, sample in enumerate(tqdm_):

            optimizer.zero_grad()
            loss = self.get_loss(sample)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % (dataloader.__len__() // 20) == 0 and step!=0:
                tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step+1, epoch_loss / (step+1+epoch*dataloader.__len__())))
                if self.writer and self.flags_obj.train_tb:
                    if self.flags_obj.verbose and step % (dataloader.__len__() // 10) == 0:
                        for name, param in self.recommender.model.named_parameters():
                            self.writer.add_histogram(name, param, step+1+epoch*dataloader.__len__())

                    self.writer.add_scalar("training_loss",
                                    epoch_loss/(step+1+epoch*dataloader.__len__()), step+1+epoch*dataloader.__len__())
        logging.info('epoch {}:  loss = {}'.format(epoch, epoch_loss/(step+1+epoch*dataloader.__len__())))

        train_loss[0] = epoch_loss

        return lr

    def get_loss(self, sample):
        
        loss = self.recommender.get_loss(sample)

        return loss

    def validate(self, epoch, total_loss):

        results = self.tester.test(total_loss=total_loss, epoch=epoch)
        self.record_metrics(epoch, results)
        print(results)
       
        return results['auc']


    

class SequenceTrainer(Trainer):
    
    def __init__(self, flags_obj, cm, dm, nc):

        super().__init__(flags_obj, cm, dm, nc)

    def set_dataloader(self):

        # training dataloader
        self.dataloader = self.recommender.dataloader
        # validation dataloader
        if self.dm.dataset_name == 'mind':
            valid_data_config = {'load_path':self.dm.load_path,
            'dataset_file': self.dm.val_dataset, 'is_sequence':True,
            'user_src_his_file':self.dm.qry_per_user,
            'user_rec_his_file':self.dm.user_his_file,
            }
            self.tester.set_dataloader(valid_data_config)

    def set_test_dataloader(self):

        if self.dm.dataset_name == 'mind':
            test_data_config = {'load_path':self.dm.load_path,
                'dataset_file': self.dm.test_dataset, 'is_sequence':True,
                'user_src_his_file':self.dm.qry_per_user,
                'user_rec_his_file':self.dm.user_his_file,
            }
            self.tester.set_dataloader(test_data_config)


    


class IV4Sequence_Trainer(SequenceTrainer):
    
    def __init__(self, flags_obj, cm, dm, nc):

        super().__init__(flags_obj, cm, dm, nc)

     
    def set_optimizer(self):
        model_optimizer = list(self.recommender.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_optimizer if 'iv_net' not in n],
             'weight_decay_rate': self.recommender.model_config['weight_decay']},
            {'params': [p for n, p in model_optimizer if 'iv_net' in n],
             'weight_decay_rate': self.recommender.model_config['l2']}
        ]


        self.optimizer = optim.Adam(params=optimizer_grouped_parameters, lr=self.recommender.model_config['lr'])
        self.s1_loss_weight = self.recommender.model_config['IV_NET']['lambda']
        self.s1_loss_weight_item = self.recommender.model_config['IV_NET']['item_IV_NET']['lambda']


    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr, train_loss):
        
        epoch_loss = train_loss[0]
        epoch_s1_loss = train_loss[1]
        epoch_s1_loss_item = train_loss[2]

        self.recommender.model.train()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        tqdm_ = tqdm(iterable=dataloader, mininterval=1, ncols=100)
        for step, sample in enumerate(tqdm_):

            loss, s1_loss, s1_loss_item = self.get_loss(sample)

            total_loss = loss + self.s1_loss_weight * s1_loss + self.s1_loss_weight_item * s1_loss_item

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_s1_loss += s1_loss.item()
            epoch_s1_loss_item += s1_loss_item.item()
            
            if step % (dataloader.__len__() // 20) == 0 and step!=0:
                tqdm_.set_description(
                    "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step+1, epoch_loss / (step+1+epoch*dataloader.__len__())))
                if self.writer and self.flags_obj.train_tb:
                    if self.flags_obj.verbose and step % (dataloader.__len__() // 10) == 0:
                        for name, param in self.recommender.model.named_parameters():
                            self.writer.add_histogram(name, param, step+1+epoch*dataloader.__len__())

                    self.writer.add_scalar("training_loss/pred_loss",
                                    epoch_loss/(step+1+epoch*dataloader.__len__()), step+1+epoch*dataloader.__len__())
                    self.writer.add_scalar("training_loss/IV_loss_user",
                                    epoch_s1_loss/(step+1+epoch*dataloader.__len__()), step+1+epoch*dataloader.__len__())
                    self.writer.add_scalar("training_loss/IV_loss_item",
                                    epoch_s1_loss_item/(step+1+epoch*dataloader.__len__()), step+1+epoch*dataloader.__len__())

        train_loss[0] = epoch_loss
        train_loss[1] = epoch_s1_loss
        train_loss[2] = epoch_s1_loss_item

        logging.info('epoch {}:  loss = {}'.format(epoch, epoch_loss/(step+1+epoch*dataloader.__len__())))

        return lr


