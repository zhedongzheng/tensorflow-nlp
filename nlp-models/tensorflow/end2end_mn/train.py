from config import args
from data import DataLoader
from model import MemoryNetwork

import tensorflow as tf
import numpy as np
import json


def main():
    train_dl = DataLoader(
        path='../dmn/temp/qa5_three-arg-relations_train.txt')
    test_dl = DataLoader(
        path='../dmn/temp/qa5_three-arg-relations_test.txt',
        is_training=False, vocab=train_dl.vocab, params=train_dl.params)

    model = MemoryNetwork(train_dl.params)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_batch = train_dl.data['size'] // args.batch_size
    for epoch in range(args.n_epochs):
        train_dl.shuffle()
        for i, batch in enumerate(train_dl.next_batch()):
            loss, acc = model.train_session(sess, batch)
            if i % args.display_step == 0:
                print("[%d/%d] | [%d/%d]" % (epoch+1, args.n_epochs, i, n_batch), end='')
                print(" | loss:%.3f | acc:%.3f" % (loss, acc))

    predicted_ids = []
    for i, batch in enumerate(test_dl.next_batch()):
        predicted_ids.append(model.predict_session(sess, batch))
    predicted_ids = np.concatenate(predicted_ids, axis=0)
    final_acc = (predicted_ids == test_dl.data['val']['answers']).mean()
    print("final testing accuracy: %.3f" % final_acc)

    demo_idx = 3
    model.demo_session(sess,
        test_dl.data['val']['inputs'][demo_idx],
        test_dl.data['val']['questions'][demo_idx],
        test_dl.data['len']['inputs_len'][demo_idx],
        test_dl.data['len']['inputs_sent_len'][demo_idx],
        test_dl.data['len']['questions_len'][demo_idx],
        test_dl.vocab['idx2word'],
        test_dl.demo,
        demo_idx)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()