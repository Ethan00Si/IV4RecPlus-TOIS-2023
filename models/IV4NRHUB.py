import torch
import torch.nn as nn
from .module import Self_Attention
import torchsnooper
from .IV_utils import IV_net
from .module import FullyConnectedLayer
mind_padding_len = 50

class IV4Rec_UI_NRHUB(nn.Module):
    def __init__(self, item_emb_matrix, rec_padding_idx, qry_emb_matrix, qry_padding_idx, config):
        
        super().__init__()

        self.rec_padding_idx = rec_padding_idx
        self.src_padding_idx = qry_padding_idx

        self.embedding_size = config['embedding_size']
        item_emb_size = config['item_dim']
        self.rec_embedding_transform = nn.Linear(self.embedding_size, item_emb_size)
        self.src_embedding_transform = nn.Linear(self.embedding_size, item_emb_size)

        self.item_embedding_layer = nn.Embedding.from_pretrained(item_emb_matrix, freeze=True)
        self.query_embedding_layer = nn.Embedding.from_pretrained(qry_emb_matrix, freeze=True)

        Dense_dim = config['Dense_dim']
        padding_len = config['history_length']
        self.search_query_att = Self_Attention(padding_len, item_emb_size, Dense_dim)
        self.browse_item_att = Self_Attention(padding_len, item_emb_size, Dense_dim)

        self.item_rep = nn.Sequential(
            nn.Linear(item_emb_size, Dense_dim),
            nn.Tanh()
        )
        self.user_rep = nn.Sequential(
           nn.Linear(config['item_dim'], Dense_dim),
           nn.Tanh(),
           Self_Attention(2, Dense_dim, 100)
        )


        self.iv_net = IV_net(qry_padding_idx, qry_emb_matrix, config['IV_NET'])
        self.loss_func = nn.BCELoss()
        self.s1_loss_func = nn.MSELoss()
        self.prob_sigmoid = nn.Sigmoid()


        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

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

        item_emb = self.item_embedding_layer(items) #batch,feature
        item_emb = self.rec_embedding_transform(item_emb)
        

        # batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(src_logs)
        query_embedding = self.src_embedding_transform(query_embedding)
        query_mask = torch.where(src_logs==self.src_padding_idx, 1, 0).bool()
        query_rep = self.search_query_att(query_embedding, query_mask)

        #batch, len, emb_dim
        browse_embedding = self.item_embedding_layer(rec_logs)
        browse_embedding = self.rec_embedding_transform(browse_embedding)
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool()
        browse_rep = self.browse_item_att(browse_embedding, browse_mask)

        iv_feats = self.iv_net(src_logs)
        s1_loss = self.s1_loss_func(iv_feats, browse_rep.detach())
        user_weight = self.user_aggregator(torch.cat([iv_feats, browse_rep],dim=-1))
        IV_user_emb = user_weight * iv_feats + (1 - user_weight) * browse_rep
        user_emb = torch.stack([IV_user_emb, query_rep], dim=1)
        user_rep = self.user_rep(user_emb)

        iv_item = self.iv_net.item_iv_forward(item_qry)
        s1_loss_item = self.s1_loss_func(iv_item, item_emb.detach())


        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_rep = item_weight * iv_item + (1 - item_weight) * item_emb
        item_rep = self.item_rep(item_rep) #batch,dense

        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)
        
        return self.loss_func(prob.squeeze(dim=-1), labels), s1_loss, s1_loss_item

    def predict(self, rec_logs, src_logs, items, item_qry):

        item_emb = self.item_embedding_layer(items) #batch,feature
        item_emb = self.rec_embedding_transform(item_emb)

        # batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(src_logs)
        query_embedding = self.src_embedding_transform(query_embedding)
        query_mask = torch.where(src_logs==self.src_padding_idx, 1, 0).bool()
        query_rep = self.search_query_att(query_embedding, query_mask)

        #batch, len, emb_dim
        browse_embedding = self.item_embedding_layer(rec_logs)
        browse_embedding = self.rec_embedding_transform(browse_embedding)
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool()
        browse_rep = self.browse_item_att(browse_embedding, browse_mask)

        iv_feats = self.iv_net(src_logs)
        user_weight = self.user_aggregator(torch.cat([iv_feats, browse_rep],dim=-1))
        IV_user_emb = user_weight * iv_feats + (1 - user_weight) * browse_rep
        user_emb = torch.stack([IV_user_emb, query_rep], dim=1)
        user_rep = self.user_rep(user_emb)

        iv_item = self.iv_net.item_iv_forward(item_qry)

        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_rep = item_weight * iv_item + (1 - item_weight) * item_emb
        item_rep = self.item_rep(item_rep) #batch,dense

        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)
        
        return prob.squeeze(dim=-1)


class IV4Rec_I_NRHUB(nn.Module):
    def __init__(self, item_emb_matrix, rec_padding_idx, qry_emb_matrix, qry_padding_idx, config):
        
        super().__init__()

        self.rec_padding_idx = rec_padding_idx
        self.src_padding_idx = qry_padding_idx

        self.embedding_size = config['embedding_size']
        item_emb_size = config['item_dim']
        self.rec_embedding_transform = nn.Linear(self.embedding_size, item_emb_size)
        self.src_embedding_transform = nn.Linear(self.embedding_size, item_emb_size)

        self.item_embedding_layer = nn.Embedding.from_pretrained(item_emb_matrix, freeze=True)
        self.query_embedding_layer = nn.Embedding.from_pretrained(qry_emb_matrix, freeze=True)

        Dense_dim = config['Dense_dim']
        padding_len = config['history_length']
        self.search_query_att = Self_Attention(padding_len, item_emb_size, Dense_dim)
        self.browse_item_att = Self_Attention(padding_len, item_emb_size, Dense_dim)

        self.item_rep = nn.Sequential(
            nn.Linear(item_emb_size, Dense_dim),
            nn.Tanh()
        )
        self.user_rep = nn.Sequential(
           nn.Linear(config['item_dim'], Dense_dim),
           nn.Tanh(),
           Self_Attention(2, Dense_dim, 100)
        )

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        self.iv_net = IV_net(qry_padding_idx, qry_emb_matrix, config['IV_NET'])
        self.loss_func = nn.BCELoss()
        self.s1_loss_func = nn.MSELoss()
        self.s1_loss_func_mask = nn.MSELoss(reduction='none')
        self.prob_sigmoid = nn.Sigmoid()


        self.item_aggregator = FullyConnectedLayer( input_size=config['item_Agg']['input_dim'],
            hidden_unit=config['item_Agg']['hid_units'], sigmoid=True
        )


    # @torchsnooper.snoop()
    def forward(self, rec_logs, src_logs, items, item_qry, labels, cor_src_logs):

        item_emb = self.item_embedding_layer(items) #batch,feature
        item_emb = self.rec_embedding_transform(item_emb)

        # batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(src_logs)
        query_embedding = self.src_embedding_transform(query_embedding)
        query_mask = torch.where(src_logs==self.src_padding_idx, 1, 0).bool()
        query_rep = self.search_query_att(query_embedding, query_mask)

        #batch, len, emb_dim
        browse_embedding = self.item_embedding_layer(rec_logs)
        browse_embedding = self.rec_embedding_transform(browse_embedding) 
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool() #used for attention
        browse_mask_for_s1_loss = torch.where(rec_logs==self.rec_padding_idx, 0, 1).bool().unsqueeze(-1).expand(-1,-1,browse_embedding.size(-1)) # for s1 loss
        # iv
        iv_feats = self.iv_net.item_iv_forward(cor_src_logs)

        s1_loss = self.s1_loss_func_mask(iv_feats, browse_embedding.detach())
        s1_loss = (s1_loss*browse_mask_for_s1_loss.float()).sum() / browse_mask_for_s1_loss.float().sum()

        user_weight = self.item_aggregator(torch.cat([iv_feats, browse_embedding],dim=-1))
        browse_embedding = user_weight * iv_feats + (1 - user_weight) * browse_embedding

        browse_rep = self.browse_item_att(browse_embedding, browse_mask)

        user_emb = torch.stack([browse_rep, query_rep], dim=1)
        # user_emb = browse_rep
        user_rep = self.user_rep(user_emb)


        iv_item = self.iv_net.item_iv_forward(item_qry)
        s1_loss_item = self.s1_loss_func(iv_item, item_emb.detach())


        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_rep = item_weight * iv_item + (1 - item_weight) * item_emb
        item_rep = self.item_rep(item_rep) #batch,dense

        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)

        return self.loss_func(prob.squeeze(dim=-1), labels), s1_loss, s1_loss_item

    def predict(self, rec_logs, src_logs, items, item_qry, cor_src_logs):
    
        item_emb = self.item_embedding_layer(items) #batch,feature
        item_emb = self.rec_embedding_transform(item_emb)

        # batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(src_logs)
        query_embedding = self.src_embedding_transform(query_embedding)
        query_mask = torch.where(src_logs==self.src_padding_idx, 1, 0).bool()
        query_rep = self.search_query_att(query_embedding, query_mask)

        #batch, len, emb_dim
        browse_embedding = self.item_embedding_layer(rec_logs)
        browse_embedding = self.rec_embedding_transform(browse_embedding) 
        browse_mask = torch.where(rec_logs==self.rec_padding_idx, 1, 0).bool() #used for attention
        # iv
        iv_feats = self.iv_net.item_iv_forward(cor_src_logs)


        user_weight = self.item_aggregator(torch.cat([iv_feats, browse_embedding],dim=-1))
        browse_embedding = user_weight * iv_feats + (1 - user_weight) * browse_embedding

        browse_rep = self.browse_item_att(browse_embedding, browse_mask)

        user_emb = torch.stack([browse_rep, query_rep], dim=1)
        user_rep = self.user_rep(user_emb)


        iv_item = self.iv_net.item_iv_forward(item_qry)


        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_rep = item_weight * iv_item + (1 - item_weight) * item_emb
        item_rep = self.item_rep(item_rep) #batch,dense

        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)

        return prob.squeeze(dim=-1)


class IV4Rec_UI__NRHUB_kuaishou(nn.Module):
    '''used on kuaishou datasets'''
    def __init__(self, item_emb_matrix, qry_emb_matrix, rec_pad, src_pad, config):
        super().__init__()

        self.rec_padding_idx = rec_pad
        self.src_padding_idx = src_pad


        self.item_transform = nn.Linear(config['embedding_size'], config['item_dim'])
        self.qry_transform = nn.Linear(config['IV_NET']['qry_dim'], config['item_dim'])

        self.item_embedding_layer = nn.Embedding.from_pretrained(item_emb_matrix, freeze=True)
        self.query_embedding_layer = nn.Embedding.from_pretrained(qry_emb_matrix, freeze=True)

        Dense_dim = config['Dense_dim']
        item_dim = config['item_dim']

        padding_len = config['history_length']
        self.search_query_att = Self_Attention(padding_len, item_dim, Dense_dim)
        self.search_click_att = Self_Attention(padding_len, item_dim, Dense_dim)
        self.browse_item_att = Self_Attention(padding_len, item_dim, Dense_dim)

        self.item_rep = nn.Sequential(
            nn.Linear(item_dim, Dense_dim),
            nn.Tanh()
        )

        self.user_rep = nn.Sequential(
           nn.Linear(item_dim, Dense_dim),
           nn.Tanh(),
           Self_Attention(3, Dense_dim, 100),
        )

        self.prob_sigmoid = nn.Sigmoid()

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        self.iv_net = IV_net(self.src_padding_idx, qry_emb_matrix, config['IV_NET'])
        self.loss_func = nn.BCELoss()
        self.s1_loss_func = nn.MSELoss()

        '''add aggregator'''
        # self.user_aggregator = Gate(config['Gate'])
        self.user_aggregator = FullyConnectedLayer(input_size = config['Agg']['input_dim'],
            hidden_unit=config['Agg']['hid_units'], sigmoid=True
        )


        self.item_aggregator = FullyConnectedLayer( input_size=config['item_Agg']['input_dim'],
            hidden_unit=config['item_Agg']['hid_units'], sigmoid=True
        )



    # @torchsnooper.snoop()
    def forward(self, browse_item, src_qry, search_click, item,  item_qry, labels):

        item_emb = self.item_embedding_layer(item) #batch,feature
        item_emb = self.item_transform(item_emb)
        item_rep = self.item_rep(item_emb) #batch,dense

        # batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(src_qry)
        query_embedding = self.qry_transform(query_embedding)
        query_mask = torch.where(src_qry==self.src_padding_idx, 1, 0).bool()
        query_rep = self.search_query_att(query_embedding, query_mask)

        click_embedding = self.item_embedding_layer(search_click)
        click_embedding = self.item_transform(click_embedding)
        search_click_mask = torch.where(search_click==self.rec_padding_idx, 1, 0).bool()
        click_rep = self.search_click_att(click_embedding, search_click_mask)

        #batch, len, emb_dim
        browse_embedding = self.item_embedding_layer(browse_item)
        browse_embedding = self.item_transform(browse_embedding)
        browse_mask = torch.where(browse_item==self.rec_padding_idx, 1, 0).bool()
        browse_rep = self.browse_item_att(browse_embedding, browse_mask)

        iv_feats = self.iv_net(src_qry)
        s1_loss = self.s1_loss_func(iv_feats, browse_rep.detach())
        user_weight = self.user_aggregator(torch.cat([iv_feats, browse_rep],dim=-1))
        IV_user_emb = user_weight * iv_feats + (1 - user_weight) * browse_rep
        user_emb = torch.stack([IV_user_emb, query_rep, click_rep], dim=1)
        user_rep = self.user_rep(user_emb)

        iv_item = self.iv_net.item_iv_forward(item_qry)
        s1_loss_item = self.s1_loss_func(iv_item, item_emb.detach())


        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_rep = item_weight * iv_item + (1 - item_weight) * item_emb
        item_rep = self.item_rep(item_rep) #batch,dense


        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)
        
        return self.loss_func(prob, labels), s1_loss, s1_loss_item

    def predict(self, browse_item, src_qry, search_click, item, item_qry):
    
        item_emb = self.item_embedding_layer(item) #batch,feature
        item_emb = self.item_transform(item_emb)
        item_rep = self.item_rep(item_emb) #batch,dense

        # batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(src_qry)
        query_embedding = self.qry_transform(query_embedding)
        query_mask = torch.where(src_qry==self.src_padding_idx, 1, 0).bool()
        query_rep = self.search_query_att(query_embedding, query_mask)

        click_embedding = self.item_embedding_layer(search_click)
        click_embedding = self.item_transform(click_embedding)
        search_click_mask = torch.where(search_click==self.rec_padding_idx, 1, 0).bool()
        click_rep = self.search_click_att(click_embedding, search_click_mask)

        #batch, len, emb_dim
        browse_embedding = self.item_embedding_layer(browse_item)
        browse_embedding = self.item_transform(browse_embedding)
        browse_mask = torch.where(browse_item==self.rec_padding_idx, 1, 0).bool()
        browse_rep = self.browse_item_att(browse_embedding, browse_mask)

        iv_feats = self.iv_net(src_qry)
        user_weight = self.user_aggregator(torch.cat([iv_feats, browse_rep],dim=-1))
        IV_user_emb = user_weight * iv_feats + (1 - user_weight) * browse_rep
        user_emb = torch.stack([IV_user_emb, query_rep, click_rep], dim=1)
        user_rep = self.user_rep(user_emb)

        iv_item = self.iv_net.item_iv_forward(item_qry)

        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_rep = item_weight * iv_item + (1 - item_weight) * item_emb
        item_rep = self.item_rep(item_rep) #batch,dense


        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)
        
        return prob

class IV4Rec_I__NRHUB_kuaishuo(nn.Module):
    '''used on kuaishou datasets'''
    def __init__(self, item_emb_matrix, qry_emb_matrix, rec_pad, src_pad, config):
        super().__init__()

        self.rec_padding_idx = rec_pad
        self.src_padding_idx = src_pad


        self.item_transform = nn.Linear(config['embedding_size'], config['item_dim'])
        self.qry_transform = nn.Linear(config['IV_NET']['qry_dim'], config['item_dim'])

        self.item_embedding_layer = nn.Embedding.from_pretrained(item_emb_matrix, freeze=True)
        self.query_embedding_layer = nn.Embedding.from_pretrained(qry_emb_matrix, freeze=True)

        Dense_dim = config['Dense_dim']
        item_dim = config['item_dim']

        padding_len = config['history_length']
        self.search_query_att = Self_Attention(padding_len, item_dim, Dense_dim)
        self.search_click_att = Self_Attention(padding_len, item_dim, Dense_dim)
        self.browse_item_att = Self_Attention(padding_len, item_dim, Dense_dim)

        self.item_rep = nn.Sequential(
            nn.Linear(item_dim, Dense_dim),
            nn.Tanh()
        )

        self.user_rep = nn.Sequential(
           nn.Linear(item_dim, Dense_dim),
           nn.Tanh(),
           Self_Attention(3, Dense_dim, 100),
        )

        self.prob_sigmoid = nn.Sigmoid()

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        self.iv_net = IV_net(self.src_padding_idx, qry_emb_matrix, config['IV_NET'])
        self.loss_func = nn.BCELoss()
        self.s1_loss_func = nn.MSELoss()
        self.s1_loss_func_mask = nn.MSELoss(reduction='none')

        '''add aggregator'''


        self.item_aggregator = FullyConnectedLayer( input_size=config['item_Agg']['input_dim'],
            hidden_unit=config['item_Agg']['hid_units'], sigmoid=True
        )



    # @torchsnooper.snoop()
    def forward(self, browse_item, src_qry, search_click, item,  item_qry, labels, cor_src_logs):

        item_emb = self.item_embedding_layer(item) #batch,feature
        item_emb = self.item_transform(item_emb)
        item_rep = self.item_rep(item_emb) #batch,dense

        # batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(src_qry)
        query_embedding = self.qry_transform(query_embedding)
        query_mask = torch.where(src_qry==self.src_padding_idx, 1, 0).bool()
        query_rep = self.search_query_att(query_embedding, query_mask)

        click_embedding = self.item_embedding_layer(search_click)
        click_embedding = self.item_transform(click_embedding)
        search_click_mask = torch.where(search_click==self.rec_padding_idx, 1, 0).bool()
        click_rep = self.search_click_att(click_embedding, search_click_mask)

        #batch, len, emb_dim
        browse_embedding = self.item_embedding_layer(browse_item)
        browse_embedding = self.item_transform(browse_embedding)
        browse_mask = torch.where(browse_item==self.rec_padding_idx, 1, 0).bool()

        browse_mask_for_s1_loss = torch.where(browse_item==self.rec_padding_idx, 0, 1).bool().unsqueeze(-1).expand(-1,-1,browse_embedding.size(-1)) # for s1 loss
        # iv
        iv_feats = self.iv_net.item_iv_forward(cor_src_logs)

        s1_loss = self.s1_loss_func_mask(iv_feats, browse_embedding.detach())
        s1_loss = (s1_loss*browse_mask_for_s1_loss.float()).sum() / browse_mask_for_s1_loss.float().sum()

        user_weight = self.item_aggregator(torch.cat([iv_feats, browse_embedding],dim=-1))
        browse_embedding = user_weight * iv_feats + (1 - user_weight) * browse_embedding
        
        browse_rep = self.browse_item_att(browse_embedding, browse_mask)


        user_emb = torch.stack([browse_rep, query_rep, click_rep], dim=1)
        user_rep = self.user_rep(user_emb)

        iv_item = self.iv_net.item_iv_forward(item_qry)
        s1_loss_item = self.s1_loss_func(iv_item, item_emb.detach())


        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_rep = item_weight * iv_item + (1 - item_weight) * item_emb
        item_rep = self.item_rep(item_rep) #batch,dense


        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)
        
        return self.loss_func(prob, labels), s1_loss, s1_loss_item

    def predict(self, browse_item, src_qry, search_click, item, item_qry, cor_src_logs):

        item_emb = self.item_embedding_layer(item) #batch,feature
        item_emb = self.item_transform(item_emb)
        item_rep = self.item_rep(item_emb) #batch,dense

        # batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(src_qry)
        query_embedding = self.qry_transform(query_embedding)
        query_mask = torch.where(src_qry==self.src_padding_idx, 1, 0).bool()
        query_rep = self.search_query_att(query_embedding, query_mask)

        click_embedding = self.item_embedding_layer(search_click)
        click_embedding = self.item_transform(click_embedding)
        search_click_mask = torch.where(search_click==self.rec_padding_idx, 1, 0).bool()
        click_rep = self.search_click_att(click_embedding, search_click_mask)

        #batch, len, emb_dim
        browse_embedding = self.item_embedding_layer(browse_item)
        browse_embedding = self.item_transform(browse_embedding)
        browse_mask = torch.where(browse_item==self.rec_padding_idx, 1, 0).bool()

        browse_mask_for_s1_loss = torch.where(browse_item==self.rec_padding_idx, 0, 1).bool().unsqueeze(-1).expand(-1,-1,browse_embedding.size(-1)) # for s1 loss
        # iv
        iv_feats = self.iv_net.item_iv_forward(cor_src_logs)


        user_weight = self.item_aggregator(torch.cat([iv_feats, browse_embedding],dim=-1))
        browse_embedding = user_weight * iv_feats + (1 - user_weight) * browse_embedding
        
        browse_rep = self.browse_item_att(browse_embedding, browse_mask)


        user_emb = torch.stack([browse_rep, query_rep, click_rep], dim=1)
        user_rep = self.user_rep(user_emb)

        iv_item = self.iv_net.item_iv_forward(item_qry)


        item_weight = self.item_aggregator(torch.cat([iv_item, item_emb], dim=-1))
        item_rep = item_weight * iv_item + (1 - item_weight) * item_emb
        item_rep = self.item_rep(item_rep) #batch,dense


        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)
        
        return prob

