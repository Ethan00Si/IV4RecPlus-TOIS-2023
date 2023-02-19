import torch.nn as nn
import torch

from .module import FullyConnectedLayer
from .IV_utils import IV_net

import torchsnooper


class IV4Rec_UI_DIN(nn.Module):
    def __init__(self, item_emb_matrix, rec_padding_idx, qry_emb_matrix, qry_padding_idx, config):
        '''
        IV4Rec+(UI)-DIN
        '''
        super().__init__()


        self.rec_padding_idx = rec_padding_idx

        self.item_emb_layer = nn.Embedding.from_pretrained(item_emb_matrix, freeze=True)
        
        self.item_layer = nn.Linear(config['input_emb_size'], config['item_dim'])


        self.attn = AttentionSequencePoolingLayer(embedding_dim=config['item_dim'])
        self.fc_layer = FullyConnectedLayer(input_size=2*config['item_dim'],
                                            hidden_unit=config['hid_units'],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='dice',
                                            dropout=config['dropout'],
                                            dice_dim=2)

        self.loss_func = nn.BCELoss()
        self.s1_loss_func = nn.MSELoss()


        self.iv_net = IV_net(qry_padding_idx, qry_emb_matrix, config['IV_NET'])

        '''add aggregator'''
        # self.user_aggregator = Gate(config['Gate'])
        self.user_aggregator = FullyConnectedLayer(input_size = config['Agg']['input_dim'],
            hidden_unit=config['Agg']['hid_units'], sigmoid=True
        )


        self.item_aggregator = FullyConnectedLayer( input_size=config['item_Agg']['input_dim'],
            hidden_unit=config['item_Agg']['hid_units'], sigmoid=True
        )


    # @torchsnooper.snoop()
    def forward(self, rec_logs, src_logs, items, item_qry, labels):
       
        # encode recommendation history 
        rec_his_emb = self.item_emb_layer(rec_logs)
        rec_his_emb = self.item_layer(rec_his_emb)
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool()

        item_emb = self.item_emb_layer(items) #batch,feature
        item_emb = self.item_layer(item_emb)

        rec_feats = self.attn(item_emb.unsqueeze(dim=1), rec_his_emb, browse_mask) 
        rec_feats = rec_feats.squeeze(dim=1) 

        # iv
        iv_feats = self.iv_net(src_logs, query=item_emb.unsqueeze(1).expand(-1, rec_logs.size(1), -1).detach())

        s1_loss = self.s1_loss_func(iv_feats, rec_feats.detach())


        # user_weight = self.user_aggregator(iv_feats, rec_feats)
        user_weight = self.user_aggregator(torch.cat([iv_feats, rec_feats],dim=-1))
        # user_weight = self.user_aggregator(torch.cat([iv_feats, rec_feats, iv_feats-rec_feats, iv_feats*rec_feats],dim=-1))
        user_emb = user_weight * iv_feats + (1 - user_weight) * rec_feats

        iv_item = self.iv_net.item_iv_forward(item_qry)
        s1_loss_item = self.s1_loss_func(iv_item, item_emb.detach())

        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_emb = item_weight * iv_item + (1 - item_weight) * item_emb
        
        concat_feature = torch.cat([item_emb, user_emb], dim=-1)

        # fully-connected layers
        output = self.fc_layer(concat_feature)

        return self.loss_func(output.squeeze(dim=-1), labels), s1_loss, s1_loss_item

    def predict(self, rec_logs, src_logs, items, item_qry):
           
         # encode recommendation history 
        rec_his_emb = self.item_emb_layer(rec_logs)
        rec_his_emb = self.item_layer(rec_his_emb)
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool()

        item_emb = self.item_emb_layer(items) #batch,feature
        item_emb = self.item_layer(item_emb)

        rec_feats = self.attn(item_emb.unsqueeze(dim=1), rec_his_emb, browse_mask) 
        rec_feats = rec_feats.squeeze(dim=1) 

        # iv
        iv_feats = self.iv_net(src_logs, query=item_emb.unsqueeze(1).expand(-1, rec_logs.size(1), -1).detach())


        user_weight = self.user_aggregator(torch.cat([iv_feats, rec_feats],dim=-1))
        # user_weight = self.user_aggregator(torch.cat([iv_feats, rec_feats, iv_feats-rec_feats, iv_feats*rec_feats],dim=-1))
        user_emb = user_weight * iv_feats + (1 - user_weight) * rec_feats

        iv_item = self.iv_net.item_iv_forward(item_qry)

        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_emb = item_weight * iv_item + (1 - item_weight) * item_emb
        
        concat_feature = torch.cat([item_emb, user_emb], dim=-1)

        # fully-connected layers
        output = self.fc_layer(concat_feature)


        return output.squeeze(dim=-1)

class IV4Rec_I_DIN(nn.Module):
    def __init__(self, item_emb_matrix, rec_padding_idx, qry_emb_matrix, qry_padding_idx, config):
        '''
        IV4Rec+(I)-DIN 
        '''
        super().__init__()


        self.rec_padding_idx = rec_padding_idx

        self.item_emb_layer = nn.Embedding.from_pretrained(item_emb_matrix, freeze=True)
        
        self.item_layer = nn.Linear(config['input_emb_size'], config['item_dim'])


        self.attn = AttentionSequencePoolingLayer(embedding_dim=config['item_dim'])
        self.fc_layer = FullyConnectedLayer(input_size=2*config['item_dim'],
                                            hidden_unit=config['hid_units'],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='dice',
                                            dropout=config['dropout'],
                                            dice_dim=2)

        self.loss_func = nn.BCELoss()
        self.s1_loss_func = nn.MSELoss()
        self.s1_loss_func_mask = nn.MSELoss(reduction='none')


        self.iv_net = IV_net(qry_padding_idx, qry_emb_matrix, config['IV_NET'])

        '''add aggregator'''
        self.item_aggregator = FullyConnectedLayer( input_size=config['item_Agg']['input_dim'],
            hidden_unit=config['item_Agg']['hid_units'], sigmoid=True
        )


    # @torchsnooper.snoop()
    def forward(self, rec_logs, src_logs, items, item_qry, labels):
       
        # encode recommendation history 
        rec_his_emb = self.item_emb_layer(rec_logs)
        rec_his_emb = self.item_layer(rec_his_emb)
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool()
        browse_mask_for_s1_loss = torch.where(rec_logs==self.rec_padding_idx, 0, 1).bool().unsqueeze(-1).expand(-1,-1,rec_his_emb.size(-1)) # for s1 loss
      
        # iv
        iv_feats = self.iv_net.item_iv_forward(src_logs)

        s1_loss = self.s1_loss_func_mask(iv_feats, rec_his_emb.detach())
        s1_loss = (s1_loss*browse_mask_for_s1_loss.float()).sum() / browse_mask_for_s1_loss.float().sum()


        user_weight = self.item_aggregator(torch.cat([iv_feats, rec_his_emb],dim=-1))
        rec_his_emb = user_weight * iv_feats + (1 - user_weight) * rec_his_emb

        item_emb = self.item_emb_layer(items) #batch,feature
        item_emb = self.item_layer(item_emb)

        rec_feats = self.attn(item_emb.unsqueeze(dim=1), rec_his_emb, browse_mask) 
        rec_feats = rec_feats.squeeze(dim=1) 



        iv_item = self.iv_net.item_iv_forward(item_qry)
        s1_loss_item = self.s1_loss_func(iv_item, item_emb.detach())

        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_emb = item_weight * iv_item + (1 - item_weight) * item_emb
        
        concat_feature = torch.cat([item_emb, rec_feats], dim=-1)

        # fully-connected layers
        output = self.fc_layer(concat_feature)

        return self.loss_func(output.squeeze(dim=-1), labels), s1_loss, s1_loss_item

    def predict(self, rec_logs, src_logs, items, item_qry):
        
        # encode recommendation history 
        rec_his_emb = self.item_emb_layer(rec_logs)
        rec_his_emb = self.item_layer(rec_his_emb)
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool()
        # iv
        iv_feats = self.iv_net.item_iv_forward(src_logs)
        iv_feats = iv_feats.masked_fill(browse_mask.unsqueeze(-1).expand(-1, -1, iv_feats.size(-1)), torch.tensor(0.0))

        user_weight = self.item_aggregator(torch.cat([iv_feats, rec_his_emb],dim=-1))
        rec_his_emb = user_weight * iv_feats + (1 - user_weight) * rec_his_emb

        item_emb = self.item_emb_layer(items) #batch,feature
        item_emb = self.item_layer(item_emb)

        rec_feats = self.attn(item_emb.unsqueeze(dim=1), rec_his_emb, browse_mask) 
        rec_feats = rec_feats.squeeze(dim=1) 

        iv_item = self.iv_net.item_iv_forward(item_qry)

        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_emb = item_weight * iv_item + (1 - item_weight) * item_emb
        
        concat_feature = torch.cat([item_emb, rec_feats], dim=-1)

        # fully-connected layers
        output = self.fc_layer(concat_feature)

        return output.squeeze(dim=-1)



class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        # TODO: DICE acitivation function
        # TODO: attention weight normalization
        self.local_att = LocalActivationUnit(hidden_unit=[64, 16], embedding_dim=embedding_dim, batch_norm=False)

    
    def forward(self, query_ad, user_behavior, mask=None):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # mask                : size -> batch_size * time_seq_len
        # output              : size -> batch_size * 1 * embedding_size
        
        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask.unsqueeze(1), torch.tensor(0))
        

        # multiply weight
        output = torch.matmul(attention_score, user_behavior)

        return output
        

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_unit=[80, 40], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_unit=hidden_unit,
                                       batch_norm=batch_norm,
                                       sigmoid=False,
                                       activation='dice',
                                       dice_dim=3)

        self.fc2 = nn.Linear(hidden_unit[-1], 1)

    # @torchsnooper.snoop()
    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        
        queries = query.expand(-1, user_behavior_len, -1)
        
        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior],
             dim=-1) # as the source code, subtraction simulates verctors' difference
        
        attention_output = self.fc1(attention_input)
        attention_score = self.fc2(attention_output) # [B, T, 1]

        return attention_score



