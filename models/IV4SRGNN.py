import time

import numpy as np
import torch
import torch.nn as nn
import torchsnooper
from .module import FullyConnectedLayer
from .IV_utils import IV_net
from torch.nn import Module, Parameter
import torch.nn.functional as F
import math
class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden



class IV4Rec_UI_SRGNN(Module):
    def __init__(self, item_emb, item_padding_idx:int, rec_padding_len, config, qry_emb, qry_padding_idx:int):
        super(IV4Rec_UI_SRGNN, self).__init__()
        self.hidden_size = config['hiddenSize']
        self.n_node = config['n_node']
        self.nonhybrid = config['nonhybrid']
        self.item_dim = config['item_dim']
        #self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.embedding = torch.nn.Embedding.from_pretrained(item_emb, freeze=True, padding_idx=item_padding_idx)
        self.embedding_transform = nn.Linear(self.item_dim, self.hidden_size, bias=True)
        self.gnn = GNN(self.hidden_size, step=config['step'])
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_output = nn.Linear(self.hidden_size * 2, 1, bias=True)
        self.reset_parameters()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_func = torch.nn.BCELoss()
        self.item_padding_idx = item_padding_idx
        self.rec_padding_len = rec_padding_len
        self.qry_padding_idx = qry_padding_idx
        self.s1_loss_func = nn.MSELoss()
        self.iv_net = IV_net(qry_padding_idx, qry_emb, config['IV_NET'])

        '''add aggregator'''
        # self.user_aggregator = Gate(config['Gate'])
        self.user_aggregator = FullyConnectedLayer(input_size = config['Agg']['input_dim'],
            hidden_unit=config['Agg']['hid_units'], sigmoid=True
        )

        self.item_aggregator = FullyConnectedLayer( input_size=config['item_Agg']['input_dim'],
            hidden_unit=config['item_Agg']['hid_units'], sigmoid=True
        )
        # self.item_emb = item_emb
        # self.qry_emb = qry_emb
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name=='embedding.weight':
                print("~~~~")
                continue
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, item_indices, src_his, item_qry):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        item_emb = self.embedding.weight[item_indices] #batch,feature
        item_emb = self.embedding_transform(item_emb)
        iv_feats = self.iv_net(src_his, query=ht.unsqueeze(1).expand(-1, src_his.size(1), -1).detach())

        s1_loss = self.s1_loss_func(iv_feats, a.detach())

        # user_weight = self.user_aggregator(iv_feats, rec_feats)
        user_weight = self.user_aggregator(torch.cat([iv_feats, a], dim=-1))
        # user_weight = self.user_aggregator(torch.cat([iv_feats, rec_feats, iv_feats-rec_feats, iv_feats*rec_feats],dim=-1))
        user_emb = user_weight * iv_feats + (1 - user_weight) * a
        iv_item = self.iv_net.item_iv_forward(item_qry)
        s1_loss_item = self.s1_loss_func(iv_item, item_emb.detach())

        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_emb = item_weight * iv_item + (1 - item_weight) * item_emb
        scores = self.sigmoid(torch.mul(user_emb, item_emb).sum(-1))
        return scores, s1_loss, s1_loss_item



    def forward(self, rec_seqs, src_seqs, item_indices, item_qry, labels, alias_inputs, A, items, mask):
        hidden = self.embedding(items)
        hidden = self.embedding_transform(hidden)
        hidden = self.gnn(A, hidden)
        seq_hidden = torch.gather(hidden, 1, alias_inputs.unsqueeze(-1).repeat(1, 1, hidden.shape[-1]))
        output, s1_loss, s1_loss_item = self.compute_scores(seq_hidden, mask, item_indices, src_seqs, item_qry)
        SRGNN_loss = self.loss_func(output.squeeze(dim=-1), labels)
        return SRGNN_loss, s1_loss, s1_loss_item

    def predict(self, rec_seqs, src_seqs, item_indices, item_qry, alias_inputs, A, items, mask):
        hidden = self.embedding(items)
        hidden = self.embedding_transform(hidden)
        hidden = self.gnn(A, hidden)
        seq_hidden = torch.gather(hidden, 1, alias_inputs.unsqueeze(-1).repeat(1, 1, hidden.shape[-1]))
        output, _ , _= self.compute_scores(seq_hidden, mask, item_indices, src_seqs, item_qry)
        return output.squeeze(dim=-1)

class IV4Rec_I_SRGNN(Module):
    def __init__(self, item_emb, item_padding_idx:int, rec_padding_len, config, qry_emb, qry_padding_idx:int):
        super(IV4Rec_I_SRGNN, self).__init__()
        self.hidden_size = config['hiddenSize']
        self.n_node = config['n_node']
        self.nonhybrid = config['nonhybrid']
        self.item_dim = config['item_dim']
        #self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.embedding = torch.nn.Embedding.from_pretrained(item_emb, freeze=True, padding_idx=item_padding_idx)
        self.embedding_transform = nn.Linear(self.item_dim, self.hidden_size, bias=True)
        self.gnn = GNN(self.hidden_size, step=config['step'])
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_output = nn.Linear(self.hidden_size * 2, 1, bias=True)
        self.reset_parameters()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_func = torch.nn.BCELoss()
        self.item_padding_idx = item_padding_idx
        self.rec_padding_len = rec_padding_len
        self.qry_padding_idx = qry_padding_idx
        self.s1_loss_func = nn.MSELoss()
        self.iv_net = IV_net(qry_padding_idx, qry_emb, config['IV_NET'])

        '''add aggregator'''
        # self.user_aggregator = Gate(config['Gate'])
        self.user_aggregator = FullyConnectedLayer(input_size = config['Agg']['input_dim'],
            hidden_unit=config['Agg']['hid_units'], sigmoid=True
        )

        self.item_aggregator = FullyConnectedLayer( input_size=config['item_Agg']['input_dim'],
            hidden_unit=config['item_Agg']['hid_units'], sigmoid=True
        )
        # self.item_emb = item_emb
        # self.qry_emb = qry_emb
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name=='embedding.weight':
                print("~~~~")
                continue
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, item_indices, src_his, item_qry):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        item_emb = self.embedding.weight[item_indices] #batch,feature
        item_emb = self.embedding_transform(item_emb)
        iv_item = self.iv_net.item_iv_forward(item_qry)
        s1_loss_item = self.s1_loss_func(iv_item, item_emb.detach())

        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_emb = item_weight * iv_item + (1 - item_weight) * item_emb
        scores = self.sigmoid(torch.mul(a, item_emb).sum(-1))
        return scores, s1_loss_item



    def forward(self, rec_seqs, src_seqs, item_indices, item_qry, labels, alias_inputs, A, items, mask):
        hidden = self.embedding(items)
        hidden = self.embedding_transform(hidden)
        hidden = self.gnn(A, hidden)
        seq_hidden = torch.gather(hidden, 1, alias_inputs.unsqueeze(-1).repeat(1, 1, hidden.shape[-1]))
        browse_mask = torch.where(rec_seqs==self.item_padding_idx, 1, 0).bool()
        # iv
        iv_feats = self.iv_net.item_iv_forward(src_seqs)
        iv_feats = iv_feats.masked_fill(browse_mask.unsqueeze(-1).expand(-1, -1, iv_feats.size(-1)), torch.tensor(0.0))

        s1_loss = self.s1_loss_func(iv_feats, seq_hidden.detach())

        user_weight = self.item_aggregator(torch.cat([iv_feats, seq_hidden],dim=-1))
        seq_hidden = user_weight * iv_feats + (1 - user_weight) * seq_hidden
        output, s1_loss_item = self.compute_scores(seq_hidden, mask, item_indices, src_seqs, item_qry)
        SRGNN_loss = self.loss_func(output.squeeze(dim=-1), labels)
        return SRGNN_loss, s1_loss, s1_loss_item

    def predict(self, rec_seqs, src_seqs, item_indices, item_qry, alias_inputs, A, items, mask):
        hidden = self.embedding(items)
        hidden = self.embedding_transform(hidden)
        hidden = self.gnn(A, hidden)
        seq_hidden = torch.gather(hidden, 1, alias_inputs.unsqueeze(-1).repeat(1, 1, hidden.shape[-1]))
        browse_mask = torch.where(rec_seqs==self.item_padding_idx, 1, 0).bool()
        # iv
        iv_feats = self.iv_net.item_iv_forward(src_seqs)
        iv_feats = iv_feats.masked_fill(browse_mask.unsqueeze(-1).expand(-1, -1, iv_feats.size(-1)), torch.tensor(0.0))
        user_weight = self.item_aggregator(torch.cat([iv_feats, seq_hidden], dim=-1))
        seq_hidden = user_weight * iv_feats + (1 - user_weight) * seq_hidden
        output, _ = self.compute_scores(seq_hidden, mask, item_indices, src_seqs, item_qry)

        return output.squeeze(dim=-1)

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

