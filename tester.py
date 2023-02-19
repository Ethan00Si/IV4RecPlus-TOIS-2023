#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from torch.utils.tensorboard import writer

from utils.metrics import judger as judge
from utils.data import Test_Dataset, SRGNN_Test_Dataset

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm


class Tester(object):

    def __init__(self, flags_obj, recommender, writer):

        self.recommender = recommender
        self.flags_obj = flags_obj
        self.judger = judge()
        self.writer = writer
        self.init_results()


    def set_dataloader(self, config):

        if self.flags_obj.dataset_name == 'mind':
            if self.flags_obj.model == 'SRGNN' or self.flags_obj.model == 'IV4Rec_UI_SRGNN' or \
                self.flags_obj.model == 'IV4Rec_I_SRGNN':
                dst = SRGNN_Test_Dataset(config, self.recommender.dm.rec_padding_idx)
            else:
                dst = Test_Dataset(config)
        else: 
            raise NotImplementedError('only support MIND dataset')

        self.dataloader = DataLoader(dst, batch_size=1024, prefetch_factor=2,\
            shuffle=False, pin_memory=False, num_workers=4)


    @torch.no_grad()
    def test(self, total_loss, epoch=0, mode='val'):
        self.recommender.model.eval()

        preds, labels = None, None
        preds, labels = self._run_eval_impression()

        if self.writer:
            self.record_loss(preds, labels,  epoch, total_loss, mode=mode)

        res = self.judger.cal_metric(preds, labels)
        self.results.update(res)
       
        return self.results

    def _run_eval_impression(self):
        """ making prediction and gather results into groups according to impression_id
        Args:

        Returns:
            preds: predicts
            labels: true labels
        """
        preds = {}
        labels = {}

        for batch_data in tqdm(iterable=self.dataloader, mininterval=1, ncols=100):
            batch_data_input, batch_label_input = batch_data[:-1],batch_data[-1]
            impression_id = batch_data_input[0]
            batch_label_input = batch_label_input.to(self.recommender.device)
            pred = self.recommender.predict(batch_data_input[1:]).squeeze(-1).tolist()
            label = batch_label_input.squeeze(-1).tolist()

            for i, ImpressionID in enumerate(impression_id):
                if  preds.__contains__(ImpressionID.item()) == False:
                    preds[ImpressionID.item()] = []
                    labels[ImpressionID.item()] = []
                preds[ImpressionID.item()].append(pred[i])
                labels[ImpressionID.item()].append(label[i])
        
        return preds, labels


    def init_results(self):

        self.results = {k: 0.0 for k in self.judger.metrics}

   
    def record_loss(self, preds, labels,  epoch, total_loss, mode='val'):

        cur_steps = 0
        for i,v in preds.items():
            loss = self.recommender.loss_func(torch.tensor(preds[i]), torch.tensor(labels[i]))
            total_loss[0] += loss.item()
            cur_steps+=1
            if self.writer:
                if mode == 'val':
                    self.writer.add_scalar("val_loss",
                                total_loss[0]/(cur_steps+epoch*len(preds)), cur_steps+epoch*len(preds))
                elif mode == 'test':
                    self.writer.add_scalar("test_loss",
                                total_loss[0]/cur_steps, cur_steps)

        return