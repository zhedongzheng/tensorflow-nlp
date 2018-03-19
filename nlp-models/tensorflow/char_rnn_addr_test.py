# -*- coding: utf-8 -*-
from char_rnn_beam import RNNTextGen
from io import open


if __name__ == '__main__':
    with open('./temp/beijing.txt', encoding='utf-8') as f:
        text = f.read()
    
    model = RNNTextGen(text, seq_len=100)
    log = model.fit()
