import torch
import torch.nn as nn
from .module import Self_Attention
import torchsnooper


class NRHUB(nn.Module):
    '''We use this variation of NRHUB on MIND dataset because no clicked news were created'''
    def __init__(self, item_emb_matrix, qry_emb_matrix, rec_pad, src_pad, qry_emb_size, config):
        '''
            "Neural News Recommendation with Heterogeneous User Behavior"
            Chuhan Wu et al. 2019 EMNLP
            without clicked news
        '''
        super(NRHUB, self).__init__()

        self.rec_padding_idx = rec_pad
        self.src_padding_idx = src_pad


        self.item_transform = nn.Linear(config['embedding_size'], config['item_emb'])
        self.qry_transform = nn.Linear(config['qry_emb_size'], config['item_emb'])

        self.item_embedding_layer = nn.Embedding.from_pretrained(item_emb_matrix, freeze=True)
        self.query_embedding_layer = nn.Embedding.from_pretrained(qry_emb_matrix, freeze=True)

        Dense_dim = config['Dense_dim']
        item_dim = config['item_emb']
        # query_feature_dim = 64
        padding_len = config['history_length']
        self.search_query_att = Self_Attention(padding_len, item_dim, Dense_dim)
        # self.search_click_att = Attention(padding_len, photo_feature_dim, Dense_dim)
        self.browse_item_att = Self_Attention(padding_len, item_dim, Dense_dim)

        self.item_rep = nn.Sequential(
            nn.Linear(item_dim, Dense_dim),
            nn.Tanh()
        )

        self.user_rep = nn.Sequential(
           nn.Linear(item_dim, Dense_dim),
           nn.Tanh(),
           Self_Attention(2, Dense_dim, 100)
        #    Attention(3, Dense_dim, 100),
        )

        self.prob_sigmoid = nn.Sigmoid()

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)


    # @torchsnooper.snoop()
    def forward(self, item, browse_item, src_qry):

        item_emb = self.item_embedding_layer(item) #batch,feature
        item_emb = self.item_transform(item_emb)
        item_rep = self.item_rep(item_emb) #batch,dense

        # batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(src_qry)
        query_embedding = self.qry_transform(query_embedding)
        query_mask = torch.where(src_qry==self.src_padding_idx, 1, 0).bool()
        query_rep = self.search_query_att(query_embedding, query_mask)

        # click_embedding = self.item_embedding_layer(search_click_photo)
        # search_click_mask = torch.where(search_click_photo==0, 1, 0).bool()
        # click_rep = self.search_click_att(click_embedding, search_click_mask)

        #batch, len, emb_dim
        browse_embedding = self.item_embedding_layer(browse_item)
        browse_embedding = self.item_transform(browse_embedding)
        browse_mask = torch.where(browse_item==self.rec_padding_idx, 1, 0).bool()
        browse_rep = self.browse_item_att(browse_embedding, browse_mask)

        user_emb = torch.stack([browse_rep, query_rep], dim=1)
        user_rep = self.user_rep(user_emb) #batch,dense

        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)
        
        return prob


class NRHUB_kuaishou(nn.Module):
    '''We use this version on two Kuaishou datasets'''
    def __init__(self, item_emb_matrix, qry_emb_matrix, rec_pad, src_pad, qry_emb_size, config):
        '''
            "Neural News Recommendation with Heterogeneous User Behavior"
            Chuhan Wu et al. 2019 EMNLP
        '''
        super().__init__()

        self.rec_padding_idx = rec_pad
        self.src_padding_idx = src_pad


        self.item_transform = nn.Linear(config['embedding_size'], config['item_emb'])
        self.qry_transform = nn.Linear(config['qry_emb_size'], config['item_emb'])

        self.item_embedding_layer = nn.Embedding.from_pretrained(item_emb_matrix, freeze=True)
        self.query_embedding_layer = nn.Embedding.from_pretrained(qry_emb_matrix, freeze=True)

        Dense_dim = config['Dense_dim']
        item_dim = config['item_emb']

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


    # @torchsnooper.snoop()
    def forward(self, item, browse_item, src_qry, search_click):

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

        user_emb = torch.stack([browse_rep, query_rep, click_rep], dim=1)
        user_rep = self.user_rep(user_emb) #batch,dense

        logits = torch.mul(item_rep, user_rep).sum(-1) #batch
        prob = self.prob_sigmoid(logits)
        
        return prob
