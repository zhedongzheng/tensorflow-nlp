from __future__ import print_function
from model import Model
from data import DataLoader

import tensorflow as tf


NUM_EPOCHS = 30

def main():
    dl = DataLoader()
    model = Model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_batch = len(dl.data['train']['Y']) // dl.batch_size
    for epoch in range(NUM_EPOCHS):
        for i, (user_id, gender_id, age_id, job_id,
            movie_id, category_ids, movie_title,
            score) in enumerate(dl.next_train_batch()):

            loss = model.train_batch(sess,
                user_id, gender_id, age_id, job_id,
                movie_id, category_ids, movie_title,
                score)
            
            if i % 500 == 0:
                print("Epoch [%d/%d] | Batch [%d/%d] | Loss: %.2f" % (
                    epoch, NUM_EPOCHS, i, n_batch, loss))
        
        losses = []
        for i, (user_id, gender_id, age_id, job_id,
            movie_id, category_ids, movie_title,
            score) in enumerate(dl.next_test_batch()):

            pred, loss = model.predict_batch(sess,
                user_id, gender_id, age_id, job_id,
                movie_id, category_ids, movie_title,
                score)
            losses.append(loss)
        print('-'*30)
        print('Testing losses:', sum(losses)/len(losses))
        print('Prediction: %.2f, Actual: %.2f' % (pred[-5], score[-5]))
        print('Prediction: %.2f, Actual: %.2f' % (pred[-4], score[-4]))
        print('Prediction: %.2f, Actual: %.2f' % (pred[-3], score[-3]))
        print('Prediction: %.2f, Actual: %.2f' % (pred[-2], score[-2]))
        print('Prediction: %.2f, Actual: %.2f' % (pred[-1], score[-1]))
        print('-'*12)


if __name__ == '__main__':
    main()
