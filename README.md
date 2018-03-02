#### Installation
```
$ pip3 install grpcio==1.9.1 tensorflow==1.6.0 sklearn scipy
```
---
#### Contents
* [Word Embedding（词向量）](https://github.com/zhedongzheng/finch#word-embedding%E8%AF%8D%E5%90%91%E9%87%8F)
* [Text Classification（文本分类）](https://github.com/zhedongzheng/finch#text-classification%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)
* [Text Generation（文本生成）](https://github.com/zhedongzheng/finch#text-generation%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90)
* [Sequence Labelling（序列标记）](https://github.com/zhedongzheng/finch#sequence-labelling%E5%BA%8F%E5%88%97%E6%A0%87%E8%AE%B0)
* [Sequence to Sequence（序列到序列）](https://github.com/zhedongzheng/finch#sequence-to-sequence%E5%BA%8F%E5%88%97%E5%88%B0%E5%BA%8F%E5%88%97)
* [Question Answering（问题回答）](https://github.com/zhedongzheng/finch#question-answering%E9%97%AE%E9%A2%98%E5%9B%9E%E7%AD%94)
---
#### Word Embedding（词向量）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_6.png" height='150'>

* Skip-Gram &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram_test.md)

* CBOW &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow_test.md)

#### Text Classification（文本分类）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_2.png" height='150'>

* CNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.md)

    * CNN + k-Max Pooling &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/kmax_conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/kmax_conv_1d_text_clf_imdb_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/kmax_conv_1d_text_clf_imdb_test.md)

* RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf_imdb_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf_imdb_test.md)

    * RNN + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_text_clf_imdb_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_text_clf_imdb_test.md)

* Only-Attention 

   * Simple Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/only_attn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/only_attn_text_clf_imdb_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/only_attn_text_clf_imdb_test.md)

   * Scaled Dot-Product Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/scaled_dot_attn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/scaled_dot_attn_text_clf_imdb_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/scaled_dot_attn_text_clf_imdb_test.md)

#### Text Generation（文本生成）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_5.png" height='150'>

* Char-RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen.py) &nbsp; | &nbsp; [English Story Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_test.py) &nbsp; | &nbsp; [Chinese Address Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_addr_test.py)
    * Char-RNN + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_beam.py) &nbsp; | &nbsp; [English Story Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_beam_test.py) &nbsp; | &nbsp; [Chinese Address Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_addr_test.py)

* CNN-Highway-RNN &nbsp; &nbsp;[Paper](https://arxiv.org/abs/1508.06615) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen.py) &nbsp; &nbsp; [PTB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen_test.py)

* Only-Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/self_attn_lm.py) &nbsp; | &nbsp; [English Story Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/self_attn_lm_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/self_attn_lm_test.md) &nbsp; | &nbsp; [Chinese Address Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/self_attn_lm_addr_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/self_attn_lm_addr_test.md)

* [Variational Autoencoder (VAE)](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/vae)
   
   * [VAE + Discriminator + Wake-Sleep](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/toward-control)

#### Sequence Labelling（序列标记）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_4.jpg" height='150'>

* Bi-RNN + CRF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_crf_clf.py) &nbsp; &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_crf_test.py) &nbsp; &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_crf_test.py) 

* Only-Attention + CRF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_clf.py) &nbsp; &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_clf_pos_test.py) &nbsp; &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_clf_chseg_test.py) 

#### Sequence to Sequence（序列到序列）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/decoration_1.png" height='150'>

* Seq2Seq &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_test.py)

   * Seq2Seq + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn_test.py) 

   * Seq2Seq + BiLSTM Encoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn_test.py) 

   * Seq2Seq + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam_test.py) 

   * Seq2Seq + BiLSTM Encoder + Attention + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate_test.py) 

* [Attention Is All You Need](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/attn_is_all_u_need)

* Pointer Network &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pointer_net.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pointer_net_test.py)

* CTC

   * [Speech Recognition](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/asr)

#### Question Answering（问题回答）
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/dmn-details.png" height='150'>

* Memory Network

   * [Dynamic Memory Network](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/dmn) 

---
#### How To Use
1. The following command clones all the files

      (it is large due to historical reasons, sorry)
   ```
   git clone https://github.com/zhedongzheng/finch.git
   ```

2. Use [contents](https://github.com/zhedongzheng/finch#contents) to find the model and test that may interest you, click on that test
   <img src="https://github.com/zhedongzheng/finch/blob/master/assets/addr_0.png" width="600">

3. Find the test file path

   <img src="https://github.com/zhedongzheng/finch/blob/master/assets/addr.png" width="600">

4. run on command line
   ```
   cd finch/nlp-models/tensorflow
   python rnn_attn_estimator_imdb_test.py
   ```
---
#### Questions Have Been Asked
* [Is the function "add_encoder_layer" in "seq2seq_ultimate.py" correct?](https://github.com/zhedongzheng/finch/issues/1)
* [in vae code,how to use LSTMcell?](https://github.com/zhedongzheng/finch/issues/2)
---
