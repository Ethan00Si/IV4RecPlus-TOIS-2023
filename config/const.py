#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

neg_sample_rate = 4


'''MIND'''
mind_path = '../data/mind'

mind_train = 'dataset/train_dataset.tsv'
mind_validation = 'dataset/test_dataset.tsv'
mind_test = 'dataset/test_dataset.tsv'

mind_train_pop = 'dataset/train_pop_dataset.tsv'
mind_validation_pop = 'dataset/test_pop_dataset.tsv'
mind_test_pop = 'dataset/test_pop_dataset.tsv'

mind_user_browse_his_file = 'user_his.json'
mind_qry_emb = 'vec/query_vec.npy'
mind_item_emb = 'vec/embedding_vec.npy'
mind_user_src_file = 'user_his.json'

mind_qry_emb_size = 768 #bert size
mind_qry_padding_idx = 0
mind_rec_padding_idx = 0

mind_rec_his_step = 50 #length of recommendation history
mind_src_his_step = 50 #length of search history

mind_batch_size = 512 #batch size


ckpt = 'ckpt'