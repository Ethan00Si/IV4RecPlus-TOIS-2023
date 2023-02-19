import torch.nn as nn
import torch

from .module import FullyConnectedLayer

import torchsnooper



class DeepInterestNetwork(nn.Module):
    def __init__(self, item_emb_matrix, rec_padding_idx, config):
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
                                            dice_dim=2)

        self.loss_func = nn.BCELoss()


    # @torchsnooper.snoop()
    def forward(self, rec_logs, items, labels):
       
        rec_his_emb = self.item_emb_layer(rec_logs)
        rec_his_emb = self.item_layer(rec_his_emb)
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool()

        item_emb = self.item_emb_layer(items) #batch,feature
        item_emb = self.item_layer(item_emb)

        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, browse_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1)], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature)

        return self.loss_func(output.squeeze(dim=-1), labels)

    def predict(self, rec_logs, items):
           
        rec_his_emb = self.item_emb_layer(rec_logs)
        rec_his_emb = self.item_layer(rec_his_emb)
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool()

        item_emb = self.item_emb_layer(items) #batch,feature
        item_emb = self.item_layer(item_emb)

        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, browse_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1)], dim=-1)
        
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



