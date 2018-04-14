<img src="https://github.com/zhedongzheng/finch/blob/master/assets/tensorflow_nlp.png" height='100'>

#### Installation 
Python 3 and Jupyter Notebook are required

CPU
```
$ pip3 install tensorflow==1.6.0 sklearn scipy tqdm
```

GPU
```
$ pip3 install tensorflow-gpu==1.6.0 sklearn scipy tqdm
```

---
#### Issues Have Been Asked
* [Is the function "add_encoder_layer" in "seq2seq_ultimate.py" correct?](https://github.com/zhedongzheng/finch/issues/1)
* [in vae code, how to use LSTMcell?](https://github.com/zhedongzheng/finch/issues/2)
---
#### Contents
* [Word Embedding（词向量）](https://github.com/zhedongzheng/finch#word-embedding%E8%AF%8D%E5%90%91%E9%87%8F)
* [Text Classification（文本分类）](https://github.com/zhedongzheng/finch#text-classification%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)
* [Text Generation（文本生成）](https://github.com/zhedongzheng/finch#text-generation%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90)
* [Text Style（文本风格）](https://github.com/zhedongzheng/finch/blob/master/README.md#text-style%E6%96%87%E6%9C%AC%E9%A3%8E%E6%A0%BC)
* [Text Matching（文本匹配）](https://github.com/zhedongzheng/finch/blob/master/README.md#text-matching%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D)
* [Speech（语音）](https://github.com/zhedongzheng/finch#speech-%E8%AF%AD%E9%9F%B3)
* [Sequence Labelling（序列标记）](https://github.com/zhedongzheng/finch#sequence-labelling%E5%BA%8F%E5%88%97%E6%A0%87%E8%AE%B0)
* [Sequence to Sequence（序列到序列）](https://github.com/zhedongzheng/finch#sequence-to-sequence%E5%BA%8F%E5%88%97%E5%88%B0%E5%BA%8F%E5%88%97)
* [Question Answering（问题回答）](https://github.com/zhedongzheng/finch/blob/master/README.md#question-answering%E9%97%AE%E9%A2%98%E5%9B%9E%E7%AD%94)
* [Recommender System（推荐系统）](https://github.com/zhedongzheng/finch/blob/master/README.md#recommender-system%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F)
* [Reinforcement Learning（强化学习）](https://github.com/zhedongzheng/finch/blob/master/README.md#reinforcement-learning%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)
* [Data Argumentation（数据增强）](https://github.com/zhedongzheng/finch/blob/master/README.md#data-argumentation%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA)
---
#### Word Embedding（词向量）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_6.png" height='100'>

* Skip-Gram &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram.py) &nbsp; &nbsp; [PTB Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram_test.ipynb)

* CBOW &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow.py) &nbsp; &nbsp; [PTB Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow_test.ipynb)

#### Text Classification（文本分类）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_2.png" height='100'>

* CNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.ipynb) &nbsp; &nbsp;

* Bi-RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf_imdb_test.ipynb)

* Self-Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/only_attn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/only_attn_text_clf_imdb_test.ipynb)

* TF-IDF + LR &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/logistic_regression.py) &nbsp; &nbsp; [IMDB Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/tfidf_imdb_test.ipynb)

#### Text Generation（文本生成）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_5.png" height='100'>

* Char-RNN + Multinomial Sampling &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen.py) &nbsp; &nbsp; [English Story Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_test.ipynb) &nbsp; &nbsp; [Chinese Address Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_addr.ipynb)

    * Char-RNN + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_beam.py) &nbsp; &nbsp; [English Story Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_beam_test.ipynb) &nbsp; &nbsp; [Chinese Address Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_addr_test.ipynb)

* Char-Aware CNN + RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen.py) &nbsp; &nbsp; [PTB Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen_test.ipynb) &nbsp; &nbsp;

* Self-Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/self_attn_lm.py) &nbsp; &nbsp; [English Story Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/self_attn_lm_test.ipynb) &nbsp; &nbsp; [Chinese Address Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/self_attn_lm_addr_test.ipynb)

* [VAE](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/vae)

#### Text Style（文本风格）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_8.png" height='100'>

* [Controllable VAE](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/toward-control)

#### Text Matching（文本匹配）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_10.jpeg" height='200'>

* [Fake News Challenge](http://www.fakenewschallenge.org/)

    * [Coursework: LR + Gradient Descent (from Scratch)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/numpy/fnc_1.ipynb)

#### Speech (语音)
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_7.png" height='100'>

* [CTC](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/asr)

#### Sequence Labelling（序列标记）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_4.jpg" height='100'>

* Bi-RNN + CRF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_crf_clf.py) &nbsp; &nbsp; [POS Tagging](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_crf_test.ipynb) &nbsp; &nbsp; [Chinese Segmentation](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_crf_test.ipynb)

* Self-Attention + CRF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_clf.py) &nbsp; &nbsp; [POS Tagging](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_clf_pos_test.ipynb) &nbsp; &nbsp;  [Chinese Segmentation](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_clf_chseg_test.ipynb)

#### Sequence to Sequence（序列到序列）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_1.png" height='100'>

* Seq2Seq &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq.py) &nbsp; &nbsp; [Sorting Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_test.ipynb)

    * Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn.py) &nbsp; &nbsp; [Sorting Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn_test.ipynb) 

    * BiLSTM Encoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn.py) &nbsp; &nbsp; [Sorting Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn_test.ipynb) 

    * Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam.py) &nbsp; &nbsp; [Sorting Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam_test.ipynb) 

    * BiLSTM Encoder + Attention + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate.py) &nbsp; &nbsp; [Sorting Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate_test.ipynb) 

* [Attention Is All You Need](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/attn_is_all_u_need)

* Pointer Network &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pointer_net.py) &nbsp; &nbsp; [Sorting Test](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pointer_net_test.ipynb)

#### Question Answering（问题回答）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/dmn-details.png" height='100'>

* [Facebook bAbI](https://research.fb.com/downloads/babi/)

    * [End-To-End Memory Network](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/end2end_mn) 

    * [Dynamic Memory Network](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/dmn) 

#### Recommender System（推荐系统）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_9.png" height='100'>

* [Hybrid Neural Network](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/movielens)

#### Reinforcement Learning（强化学习）
```pip3 install gym ```
* Policy Gradient &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rl-models/pg.py) &nbsp; &nbsp; [CartPole Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rl-models/pg_cartpole_test.py)

#### Data Argumentation（数据增强）
* [Bi-Translation](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/other/bi_translation.ipynb)
