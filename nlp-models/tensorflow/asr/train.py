from data_loader import DataLoader
from model import Model

import tensorflow as tf
import numpy as np


def main():
    num_epochs = 20
    batch_size = 30
    
    dl = DataLoader(batch_size=batch_size)
    model = Model(dl.num_classes)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_batch = len(dl.inputs) // batch_size
    for epoch in range(1, num_epochs+1):
        for i, (inputs, seq_lens, sparse_targets) in enumerate(dl.next_batch()):
            loss, edit_dist = model.train_batch(sess, inputs, seq_lens, sparse_targets)
            print("Epoch [%d/%d] | Batch [%d/%d] | Loss:%.3f | Edit Dist: %.3f |" % (
                    epoch, num_epochs, i, n_batch, loss, edit_dist))
            
    preds = model.test_batch(sess, dl.inputs_val, dl.seq_lens_val)
    print('Prediction:', ''.join([dl.idx2char[idx] for idx in preds[0]]))
    print('Actual:', ''.join([dl.idx2char[idx] for idx in dl.targets_val]))

if __name__ == '__main__':
    main()
