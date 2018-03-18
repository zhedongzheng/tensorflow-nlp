from __future__ import print_function

import paddle.v2 as paddle
import pandas as pd
import json


word2idx = paddle.dataset.movielens.get_movie_title_dict()
idx2word = {i+1 : w for w, i in word2idx.items()}
idx2word['<pad>'] = 0

with open('vocab.txt', 'w') as outfile:
    json.dump(idx2word, outfile, ensure_ascii=False)


user_info = paddle.dataset.movielens.user_info()
movie_info = paddle.dataset.movielens.movie_info()
train_set_creator = paddle.dataset.movielens.train()
test_set_creator = paddle.dataset.movielens.test()


def fn(creator):
    train_df = {'user_id': [], 'gender_id': [], 'age_id': [], 'job_id': [], 'movie_id': [],
                'category_ids': [], 'movie_title': [], 'score': []}
    for i, train_sample in enumerate(creator()):
        uid = train_sample[0]
        mov_id = train_sample[len(user_info[uid].value())]
        mov_dict = movie_info[mov_id].__dict__
        train_df['user_id'].append(train_sample[0])
        train_df['gender_id'].append(train_sample[1])
        train_df['age_id'].append(train_sample[2])
        train_df['job_id'].append(train_sample[3])
        train_df['movie_id'].append(train_sample[4])

        category_ids = [0]*18
        for c_id in train_sample[5]:
            category_ids[c_id] = 1
        category_ids = [str(id) for id in category_ids]
        train_df['category_ids'].append(' '.join(category_ids))

        movie_title_idx = [idx + 1 for idx in train_sample[6]]
        movie_title_idx = [str(id) for id in movie_title_idx]
        train_df['movie_title'].append(' '.join(movie_title_idx))

        train_df['score'].append(train_sample[7][0])
    return train_df
    print(i)


train_df = pd.DataFrame(fn(train_set_creator))
train_df.to_csv('./movielens_train.csv', index=False)

test_df = pd.DataFrame(fn(test_set_creator))
test_df.to_csv('./movielens_test.csv', index=False)
