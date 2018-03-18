from __future__ import print_function
from tqdm import tqdm

import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, batch_size=256):
        self.data = {
            'train': {
                'X': {
                    'user_id': None,
                    'gender_id': None,
                    'age_id': None,
                    'job_id': None,
                    'movie_id': None,
                    'category_ids': None,
                    'movie_title': None,
                },
                'Y': None,
            },
            'test': {
                'X': {
                    'user_id': None,
                    'gender_id': None,
                    'age_id': None,
                    'job_id': None,
                    'movie_id': None,
                    'category_ids': None,
                    'movie_title': None,
                },
                'Y': None,
            }
        }
        self.batch_size = batch_size
        self.process('train')
        self.process('test')

    def process(self, routine):
        csv = pd.read_csv('./data/movielens_%s.csv'%routine)

        self.data[routine]['X']['user_id'] = csv['user_id'].values
        self.data[routine]['X']['gender_id'] = csv['gender_id'].values
        self.data[routine]['X']['age_id'] = csv['age_id'].values
        self.data[routine]['X']['job_id'] = csv['job_id'].values
        self.data[routine]['X']['movie_id'] = csv['movie_id'].values

        self.data[routine]['X']['category_ids'] = []
        for category_id in tqdm(
            csv['category_ids'].values, total=len(csv), ncols=70):
            self.data[routine]['X']['category_ids'].append([int(i) for i in category_id.split()])
        self.data[routine]['X']['category_ids'] = np.array(self.data[routine]['X']['category_ids'])

        self.data[routine]['X']['movie_title'] = []
        for mov_title in tqdm(
            csv['movie_title'].values, total=len(csv), ncols=70):
            temp_li = [0] * 10
            mov_title_li = mov_title.split()
            for i in range(len(mov_title_li[:10])):       
                temp_li[i] = int(mov_title_li[i])
            self.data[routine]['X']['movie_title'].append(temp_li)
        self.data[routine]['X']['movie_title'] = np.array(self.data[routine]['X']['movie_title'])

        self.data[routine]['Y'] = csv['score'].values
    
    def next_train_batch(self):
        for i in range(0, len(self.data['train']['X']['user_id']), self.batch_size):
            yield (self.data['train']['X']['user_id'][i : i+self.batch_size],
                   self.data['train']['X']['gender_id'][i : i+self.batch_size],
                   self.data['train']['X']['age_id'][i : i+self.batch_size],
                   self.data['train']['X']['job_id'][i : i+self.batch_size],
                   self.data['train']['X']['movie_id'][i : i+self.batch_size],
                   self.data['train']['X']['category_ids'][i : i+self.batch_size],
                   self.data['train']['X']['movie_title'][i : i+self.batch_size],
                   self.data['train']['Y'][i : i+self.batch_size])

    def next_test_batch(self):
        for i in range(0, len(self.data['test']['X']['user_id']), self.batch_size):
            yield (self.data['test']['X']['user_id'][i : i+self.batch_size],
                   self.data['test']['X']['gender_id'][i : i+self.batch_size],
                   self.data['test']['X']['age_id'][i : i+self.batch_size],
                   self.data['test']['X']['job_id'][i : i+self.batch_size],
                   self.data['test']['X']['movie_id'][i : i+self.batch_size],
                   self.data['test']['X']['category_ids'][i : i+self.batch_size],
                   self.data['test']['X']['movie_title'][i : i+self.batch_size],
                   self.data['test']['Y'][i : i+self.batch_size])
