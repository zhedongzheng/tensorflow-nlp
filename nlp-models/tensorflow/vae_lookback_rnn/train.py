from __future__ import print_function
from data import IMDB
from model import VRAE
from config import args
import json
import tensorflow as tf


def main():
    dataloader = IMDB()
    params = {
        'vocab_size': len(dataloader.word2idx),
        'word2idx': dataloader.word2idx,
        'idx2word': dataloader.idx2word,}
    print('Vocab Size:', params['vocab_size'])
    model = VRAE(params)
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(args.num_epoch):
        dataloader.update_word_dropout()
        print("\nWord Dropout")
        dataloader.shuffle()
        print("Data Shuffled", end='\n\n')
        for i, (enc_inp, dec_inp, dec_out) in enumerate(dataloader.next_batch()):
            log = model.train_session(sess, enc_inp, dec_inp, dec_out)
            if i % args.display_loss_step == 0:
                print("Step %d | [%d/%d] | [%d/%d]" % (log['step'], epoch+1, args.num_epoch, i, len(dataloader.enc_inp)//args.batch_size), end='')
                print(" | nll_loss:%.1f | kl_w:%.3f | kl_loss:%.2f \n" % (log['nll_loss'], log['kl_w'], log['kl_loss']))
        
        model.generate(sess)
        model.reconstruct(sess, enc_inp[-1], dec_inp[-1])
        model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
        model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it')
        
        save_path = saver.save(sess, './saved/vrae.ckpt')
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()