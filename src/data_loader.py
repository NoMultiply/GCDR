import os
import random
from collections import defaultdict

import torch
from logger import DATA_ROOT, logger
from tqdm import tqdm


class Dataset:
    def __init__(self, dataset, device=None, max_history_length=5, min_history_length=5):
        self.dataset = dataset
        self.device = device

        self.user_map = {}

        self.item_map = {}
        self.item_features = []
        self.item_categories = []
        self.all_catetories = set()
        with open(os.path.join(DATA_ROOT, dataset, 'item_categories.txt')) as fin:
            for line in tqdm(fin, desc=f'[{dataset}][item_categories.txt]'):
                iid, categories = line.strip().split('\t')
                self.item_map[iid] = len(self.item_map)
                categories = set(categories.split('|'))
                self.item_categories.append(categories)
                self.all_catetories = self.all_catetories.union(categories)
        self.n_items = len(self.item_map)

        self.train_data, self.valid_data, self.test_data = {}, {}, {}
        self.user_categories = defaultdict(set)
        for split in ('train', 'valid', 'test'):
            data = {}
            n_skip = 0
            with open(os.path.join(DATA_ROOT, dataset, f'{split}.txt')) as fin:
                for line in tqdm(fin, desc=f'[{dataset}][{split}.txt]'):
                    uid, iids = line.strip('\n').split('\t')
                    if iids == '':
                        n_skip += 1
                        continue
                    self.user_map[uid] = self.user_map.get(uid, len(self.user_map))
                    data[self.user_map[uid]] = [self.item_map[iid] for iid in iids.split(',')]
                    for iid in data[self.user_map[uid]]:
                        self.user_categories[self.user_map[uid]].update(self.item_categories[iid])
            assert n_skip == 0, f'[WARNING][{split}] Skip {n_skip} users due to lack of positive or negative interactions or no features.'
            setattr(self, split + '_data', data)

        
        self.n_users = len(self.user_map)

        for split in ('valid', 'test'):
            data = getattr(self, split + '_data')
            for k in data:
                data[k] = torch.LongTensor(data[k]).to(device)

        self.train_data_tensor = {}
        for k in self.train_data:
            self.train_data_tensor[k] = torch.LongTensor(self.train_data[k]).to(device)

        self.seq_train_data = []
        self.seq_test_data = {}
        for uid, pos_iids in self.train_data.items():
            histories = []
            for iid in pos_iids:
                if len(histories) >= min_history_length:
                    len_seq = len(histories[-max_history_length:])
                    seq = [self.n_items] * (max_history_length - len_seq) + histories[-max_history_length:]
                    # seq = histories[-max_history_length:] + [self.n_items] * (max_history_length - len_seq)
                    self.seq_train_data.append((uid, seq, len_seq, iid))
                histories.append(iid)
            len_seq = len(histories[-max_history_length:])
            seq = [self.n_items] * (max_history_length - len_seq) + histories[-max_history_length:]
            # seq = histories[-max_history_length:] + [self.n_items] * (max_history_length - len_seq)
            self.seq_test_data[uid] = (seq, len_seq)
        self.max_history_length = max_history_length
        self.cat2id = {c: i for i, c in enumerate(self.all_catetories)}
        self.user_cat_hist = None

    def calc_user_cat_hist(self):
        self.user_cat_hist = []
        for _, pos_iids in tqdm(self.train_data.items(), bar_format='{l_bar}{r_bar}', desc='[calc_user_cat_hist]'):
            hist = [0] * len(self.cat2id)
            for iid in pos_iids:
                for c in self.item_categories[iid]:
                    hist[self.cat2id[c]] += 1
            self.user_cat_hist.append(hist)
        self.user_cat_hist = torch.FloatTensor(self.user_cat_hist).to(self.device)
        self.user_cat_hist = self.user_cat_hist / self.user_cat_hist.sum(dim=-1, keepdim=True)

        self.item_cat_one_hot = []
        for iid in tqdm(range(self.n_items), bar_format='{l_bar}{r_bar}', desc='[item_cat_one_hot]'):
            one_hot = [0] * len(self.cat2id)
            for c in self.item_categories[iid]:
                one_hot[self.cat2id[c]] += 1
            self.item_cat_one_hot.append(hist)
        self.item_cat_one_hot = torch.FloatTensor(self.item_cat_one_hot).to(self.device)
        
    def __repr__(self):
        string = ''
        string += '-------------------------------------\n'
        string += f'[{self.dataset}]\n'
        string += '\n'

        string += f'# of users: {self.n_users}\n'
        string += f'# of items: {self.n_items}\n'
        string += f'# of nodes: {self.n_users} + {self.n_items} = {self.n_users + self.n_items}\n'
        string += '\n'

        n_train = sum(len(x) for x in self.train_data.values())
        string += f'# of training interactions: {n_train}\n'

        n_valid = sum(len(x) for x in self.valid_data.values())
        string += f'# of valid interactions: {n_valid}\n'

        n_test = sum(len(x) for x in self.test_data.values())
        string += f'# of testing interactions: {n_test}\n'

        n_interactions = n_train + n_valid + n_test
        string += f'# of interactions: {n_interactions}\n'


        if self.seq_train_data is not None:
            string += f'# of seq. training data: {len(self.seq_train_data)}\n'

        string += '\n'
        string += f'% Density: {n_interactions / (self.n_users * self.n_items) * 100}\n' 

        string += '\n'
        string += f'# Categories: {len(self.all_catetories)}\n' 
        string += '-------------------------------------'

        return string



def main():
    logger.print(Dataset('movielens'))
    logger.print(Dataset('kuairec'))
    logger.print(Dataset('yelp'))
    logger.print(Dataset('books'))

    # dataset = Dataset('movielens')
    # dataset.calc_user_cat_hist()
    # logger.print(dataset)


if __name__ == '__main__':
    main()
