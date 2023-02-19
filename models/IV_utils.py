import torch
import torch.nn as nn

import torchsnooper
from models.module import FullyConnectedLayer




class Self_Additive_Attention(nn.Module):
    def __init__(self, seq_length, query_input_dim, value_input_dim, Dense_dim, output_dim):
        super().__init__()
        self.seq_length = seq_length
        self.q_input_dim = query_input_dim
        self.k_input_dim = value_input_dim
        self.dense_dim = Dense_dim
        self.Wk = nn.Linear(value_input_dim, self.dense_dim, bias=False)
        self.Wq = nn.Linear(self.q_input_dim, self.dense_dim, bias=False)
        self.Wv = nn.Linear(value_input_dim, output_dim, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.vq = nn.Parameter(torch.randn(1,1,self.dense_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.vq)
        nn.init.xavier_normal_(self.Wq.weight)
        nn.init.xavier_normal_(self.Wk.weight)
        nn.init.xavier_normal_(self.Wv.weight)
    
    def forward(self, value, query=None, mask=None):
        """
        Args:
            x: [bs, sl, hd]
        Return
            res: [bs, od]
        """
        # bs, sl ,dd
        key = None
        if query is None:
            key = self.tanh(self.Wk(value))
        else:
            key = self.tanh(self.Wq(query) + self.Wk(value))

        # bs, 1, sl
        score = self.vq.matmul(key.transpose(-2,-1))
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(-2), torch.tensor(-1e6))
        
        weight = self.softmax(score)
        res = weight.matmul(self.Wv(value)).sum(dim=-2)
        # bs, hd
        return res

class IV_net(nn.Module):
    def __init__(self, padding_idx, qry_emb, config):
        '''multi-head self attention + additive attention'''
        super().__init__()

        self.padding_idx = padding_idx
        self.qry_emb = torch.nn.Embedding.from_pretrained(qry_emb, freeze=True, padding_idx=padding_idx)
        
        '''user branch'''
        self.linear = nn.Linear(config['qry_dim'], config['rec_item_dim'])

        self.multi_head_self_att = nn.MultiheadAttention(embed_dim=config['rec_item_dim'], 
            num_heads=config['num_heads'], batch_first=True
        )

        self.add_att = Self_Additive_Attention(config['src_his_step'],  config['rec_item_dim'], config['rec_item_dim'], config['dense'], config['rec_item_dim'])
        self.mlp = FullyConnectedLayer(input_size=config['rec_item_dim'], hidden_unit=config['hid_units'],
            dropout=config['dropout'])

        '''item branch'''
        self.item_mlp = FullyConnectedLayer( input_size = config['qry_dim'], 
            hidden_unit= config['item_IV_NET']['hid_units'], sigmoid=False, dropout=config['item_IV_NET']['dropout']
        )

    def forward(self, qry_seqs, query=None):
        '''regress search history on recommendation history'''
    
        seqs = self.qry_emb(qry_seqs)
        seqs = self.linear(seqs)
        timeline_mask = torch.where(qry_seqs==self.padding_idx, 1, 0).bool()

        output, _ = self.multi_head_self_att(query = seqs, key = seqs, value = seqs, key_padding_mask = timeline_mask )
        output = self.add_att(value=output, query=query, mask = timeline_mask)
        output = self.mlp(output)
        return output

    def item_iv_forward(self, item_qry):
        '''regress queries on items'''

        qry_emb = self.qry_emb(item_qry)
        output = self.item_mlp(qry_emb)

        return output


