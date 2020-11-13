		Code has been run on Google Colab, thanks Google for providing computational resources

#### Contents

* Natural Language Processing（自然语言处理）

	* [Text Classification（文本分类）](https://github.com/zhedongzheng/finch#text-classification)
	
		* IMDB（ENG）

		* CLUE Emotion Analysis Dataset (CHN)

	* [Text Matching（文本匹配）](https://github.com/zhedongzheng/finch#text-matching)

		* SNLI（ENG）
		
		* 微众银行智能客服（CHN）

		* 蚂蚁金融语义相似度 (CHN)

	* [Intent Detection and Slot Filling（意图检测与槽位填充）](https://github.com/zhedongzheng/finch#intent-detection-and-slot-filling)

		* ATIS（ENG）

	* [Retrieval Dialog（检索式对话）](https://github.com/zhedongzheng/finch#retrieval-dialog)

		* ElasticSearch

			* Sparse Retrieval

			* Dense Retrieval

	* [Generative Dialog（生成式对话）](https://github.com/zhedongzheng/finch#generative-dialog)

		* Large-scale Chinese Conversation Dataset (CHN)

	* [Multi-turn Dialogue Rewriting（多轮对话 的 指代消歧、省略补全）](https://github.com/zhedongzheng/finch#multi-turn-dialogue-rewriting)

		* 20k 腾讯 AI 研发数据（CHN）

	* [Semantic Parsing（语义解析）](https://github.com/zhedongzheng/finch#semantic-parsing)
	
		* Facebook's Hierarchical Task Oriented Dialog（ENG）
	
	* [Multi-hop Question Answering（多跳问题回答）](https://github.com/zhedongzheng/finch#multi-hop-question-answering)
	
		* bAbI（ENG）
		
	* [Text Processing Tools（文本处理工具）](https://github.com/zhedongzheng/finch#text-processing-tools)

* Knowledge Graph（知识图谱）

	* [Knowledge Graph Completion（知识图谱补全）](https://github.com/zhedongzheng/finch#knowledge-graph-completion)
	
	* [Knowledge Base Question Answering（知识图谱问答）](https://github.com/zhedongzheng/finch#knowledge-base-question-answering)
	
	* [Knowledge Graph Tools（知识图谱工具）](https://github.com/zhedongzheng/finch#knowledge-graph-tools)

* [Recommender System（推荐系统）](https://github.com/zhedongzheng/finch#recommender-system)

	* Movielens 1M（English Data）

---

## Text Classification

```
└── finch/tensorflow2/text_classification/imdb
	│
	├── data
	│   └── glove.840B.300d.txt          # pretrained embedding, download and put here
	│   └── make_data.ipynb              # step 1. make data and vocab: train.txt, test.txt, word.txt
	│   └── train.txt  		     # incomplete sample, format <label, text> separated by \t 
	│   └── test.txt   		     # incomplete sample, format <label, text> separated by \t
	│   └── train_bt_part1.txt  	     # (back-translated) incomplete sample, format <label, text> separated by \t
	│
	├── vocab
	│   └── word.txt                     # incomplete sample, list of words in vocabulary
	│	
	└── main
		└── sliced_rnn.ipynb         # step 2: train and evaluate model
		└── ...
```

* Task: [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/)（English Data）
	
        Training Data: 25000, Testing Data: 25000, Labels: 2
	
	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/train.txt)
		
		* [\<Text File>: Data Example (Back-Translated)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/train_bt_part1.txt)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/vocab/word.txt)

	* Model: TF-IDF + Logistic Regression ([Sklearn](https://scikit-learn.org/stable/))

		| Logistic Regression | Binary TF | NGram Range | Knowledge Dist | Testing Accuracy |
		| --- | --- | --- | --- | --- |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/text_classification/imdb/tfidf_lr_binary_false.ipynb) | False | (1, 1) | False | 88.3% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/text_classification/imdb/tfidf_lr_binary_true.ipynb) | True | (1, 1) | False | 88.8% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/sklearn/text_classification/imdb/tfidf_lr_binary_true_bigram.ipynb) | True | (1, 2) | False | 89.6% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/lr_know_dist.ipynb) | True | (1, 2) | True | 90.7% |

		-> [PySpark](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/spark/text_classification/imdb/tfidf_lr.ipynb) Equivalent

	* Model: [FastText](https://arxiv.org/abs/1607.01759)
	
		| FastText | Setting | Testing Accuracy |
		| --- | --- | --- |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/unigram.ipynb) | Unigram | 87.3% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/bigram.ipynb) | Bigram | 89.8% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/autotune.ipynb) | Autotune | 90.1% |
	
	```
	Back-Translation increases training data from 25000 to 50000

	which is done by "english -> french -> english" translation
	```

	```python
	from googletrans import Translator

	translator = Translator()

	translated = translator.translate(text, src='en', dest='fr').text

	back = translator.translate(translated, src='fr', dest='en').text
	```
	
	* Model: [TextCNN](https://arxiv.org/abs/1408.5882)

		* TensorFlow 2

			* [\<Notebook> CNN + Attention](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/cnn_attention_bt_char_label_smooth_cyclical.ipynb)
			
				-> 91.8 % Testing Accuracy

	* Model: [Sliced RNN](https://arxiv.org/abs/1807.02291)

		* TensorFlow 2

			* [\<Notebook> Sliced LSTM](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn_bt_char_label_smooth_clr.ipynb)
			
				-> 92.6 % Testing Accuracy

				This result (without transfer learning) is higher than [CoVe](https://arxiv.org/pdf/1708.00107.pdf) (with transfer learning)

	* Model: [BERT](https://arxiv.org/abs/1810.04805)

		* TensorFlow 2 + [transformers](https://github.com/huggingface/transformers)

			| Bert (base-uncased) | Batch Size | Max Length | Testing Accuracy |
			| --- | --- | --- | --- |
			| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/bert_finetune_32_128.ipynb) | 32 | 128 | 92.6% |
			| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/bert_finetune_16_200.ipynb) | 16 | 200 | 93.3% |
			| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/bert_finetune_12_256.ipynb) | 12 | 256 | 93.8% |
			| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/bert_finetune_8_300.ipynb) | 8 | 300 | 94% |

	* Model: [RoBERTa](https://arxiv.org/abs/1907.11692)

		* TensorFlow 2 + [transformers](https://github.com/huggingface/transformers)

			* [\<Notebook> RoBERTa (base) { batch_size=8, max_len=300 }](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/roberta_finetune_8_300.ipynb)
			
			 	-> 94.7% Testing Accuracy


```
└── finch/tensorflow2/text_classification/clue
	│
	├── data
	│   └── make_data.ipynb              # step 1. make data and vocab
	│   └── train.txt  		     # download from clue benchmark
	│   └── test.txt   		     # download from clue benchmark
	│
	├── vocab
	│   └── label.txt                    # list of emotion labels
	│	
	└── main
		└── bert_finetune.ipynb      # step 2: train and evaluate model
		└── ...
```

* Task: [CLUE Emotion Analysis Dataset](https://github.com/CLUEbenchmark/CLUEmotionAnalysis2020)（Chinese Data）
	
        Training Data: 31728, Testing Data: 3967, Labels: 7

	* Model: TF-IDF + Linear Model

		| Logistic Regression | Binary TF | NGram Range | Split By | Testing Accuracy |
		| --- | --- | --- | --- | --- |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/clue/main/tfidf_lr_binary_f_char_unigram.ipynb) | False | (1, 1) | Char | 57.4% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/clue/main/tfidf_lr_binary_t_word_unigram.ipynb) | True | (1, 1) | Word | 57.7% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/clue/main/tfidf_lr_binary_f_word_bigram.ipynb) | False | (1, 2) | Word | 57.8% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/clue/main/tfidf_lr_binary_f_word_unigram.ipynb) | False | (1, 1) | Word | 58.3% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/clue/main/tfidf_lr_binary_t_char_bigram.ipynb) | True | (1, 2) | Char | 59.1% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/clue/main/tfidf_lr_binary_f_char_bigram.ipynb) | False | (1, 2) | Char | 59.4% |

	* Model: Deep Model
				
		| Code | Model | Env | Testing Accuracy |
		| --- | --- | --- | --- |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/clue/main/bert_finetune.ipynb) | [BERT](https://arxiv.org/abs/1810.04805) | TF2 | 61.7% |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/clue/main/bert_further_pretrain_finetune.ipynb) | BERT + [TAPT](https://arxiv.org/abs/2004.10964) ([\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/clue/main/bert_further_pretrain.ipynb)) | TF2 | 62.3% |

---

## Text Matching

```
└── finch/tensorflow2/text_matching/snli
	│
	├── data
	│   └── glove.840B.300d.txt       # pretrained embedding, download and put here
	│   └── download_data.ipynb       # step 1. run this to download snli dataset
	│   └── make_data.ipynb           # step 2. run this to generate train.txt, test.txt, word.txt 
	│   └── train.txt  		  # incomplete sample, format <label, text1, text2> separated by \t 
	│   └── test.txt   		  # incomplete sample, format <label, text1, text2> separated by \t
	│
	├── vocab
	│   └── word.txt                  # incomplete sample, list of words in vocabulary
	│	
	└── main              
		└── dam.ipynb      	  # step 3. train and evaluate model
		└── esim.ipynb      	  # step 3. train and evaluate model
		└── ......
```

* Task: [SNLI](https://nlp.stanford.edu/projects/snli/)（English Data）

        Training Data: 550152, Testing Data: 10000, Labels: 3

	* [\<Notebook>: Download Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/download_data.ipynb)

	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/make_data.ipynb)

		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/train.txt)

		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/vocab/word.txt)
		  
	| Code | Reference | Env | Testing Accuracy |
	| --- | --- | --- | --- |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/dam.ipynb) | [DAM](https://arxiv.org/abs/1606.01933) | TF2 | 85.3% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/pyramid_multi_attn.ipynb) | [Match Pyramid](https://arxiv.org/abs/1602.06359) | TF2 | 87.1% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/esim.ipynb) | [ESIM](https://arxiv.org/abs/1609.06038) | TF2 | 87.4% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/re2_birnn.ipynb) | [RE2](https://arxiv.org/abs/1908.00300) | TF2 | 87.7% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/re2_3_birnn_label_smooth.ipynb) | RE3 | TF2 | 88.3% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/bert_finetune.ipynb) | [BERT](https://arxiv.org/abs/1810.04805) | TF2 | 90.4% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/roberta_finetune.ipynb) | [RoBERTa](https://arxiv.org/abs/1907.11692) | TF2 | 91.1% |

```
└── finch/tensorflow2/text_matching/chinese
	│
	├── data
	│   └── make_data.ipynb           # step 1. run this to generate char.txt and char.npy
	│   └── train.csv  		  # incomplete sample, format <text1, text2, label> separated by comma 
	│   └── test.csv   		  # incomplete sample, format <text1, text2, label> separated by comma
	│
	├── vocab
	│   └── cc.zh.300.vec             # pretrained embedding, download and put here
	│   └── char.txt                  # incomplete sample, list of chinese characters
	│   └── char.npy                  # saved pretrained embedding matrix for this task
	│	
	└── main              
		└── pyramid.ipynb      	  # step 2. train and evaluate model
		└── esim.ipynb      	  # step 2. train and evaluate model
		└── ......
```

* Task: [微众银行智能客服](https://github.com/terrifyzhao/text_matching/tree/master/input)（Chinese Data）

        Training Data: 100000, Testing Data: 10000, Labels: 2, Balanced

	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/data/train.csv) (数据示例)
		
		* [\<Text File>: Vocabulary](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/vocab/char.txt)
		
	* Model	(can be compared to [this benchmark](https://github.com/wangle1218/deep_text_matching) since the dataset is the same)

	| Code | Reference | Env | Split by | Testing Accuracy |
	| --- | --- | --- | --- | --- |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/main/word_re2_cyclical_label_smooth.ipynb) | [RE2](https://arxiv.org/abs/1908.00300) | TF2 | Word | 82.5% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/main/esim.ipynb) | [ESIM](https://arxiv.org/abs/1609.06038) | TF2 | Char | 82.5% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/main/pyramid.ipynb) | [Match Pyramid](https://arxiv.org/abs/1602.06359) | TF2 | Char | 82.7% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/main/re2_cyclical_label_smooth.ipynb) | [RE2](https://arxiv.org/abs/1908.00300) | TF2 | Char | 83.8% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/main/bert_finetune.ipynb) | [BERT](https://arxiv.org/abs/1810.04805) | TF2 | Char | 83.8% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/text_matching/chinese/main/bert_chinese_wwm.ipynb) | [BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) | TF1 + [bert4keras](https://github.com/bojone/bert4keras) | Char | 84.75% |

```
└── finch/tensorflow2/text_matching/ant
	│
	├── data
	│   └── make_data.ipynb           # step 1. run this to generate char.txt and char.npy
	│   └── train.json           	  # incomplete sample, format <text1, text2, label> separated by comma 
	│   └── dev.json   		  # incomplete sample, format <text1, text2, label> separated by comma
	│
	├── vocab
	│   └── cc.zh.300.vec             # pretrained embedding, download and put here
	│   └── char.txt                  # incomplete sample, list of chinese characters
	│   └── char.npy                  # saved pretrained embedding matrix for this task
	│	
	└── main              
		└── pyramid.ipynb      	  # step 2. train and evaluate model
		└── bert.ipynb      	  # step 2. train and evaluate model
		└── ......
```

* Task: [蚂蚁金融语义相似度](https://cluebenchmarks.com/introduce.html)（Chinese Data）

        Training Data: 34334, Testing Data: 4316, Labels: 2, Imbalanced

	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/ant/data/make_data.ipynb)

		* [\<Text File>: Vocabulary](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/ant/vocab/char.txt)

	* Model

	| Code | Reference | Env | Split by | Testing Accuracy |
	| --- | --- | --- | --- | --- |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/ant/main/re2.ipynb) | [RE2](https://arxiv.org/abs/1908.00300) | TF2 | Char | 66.5% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/ant/main/pyramid.ipynb) | [Match Pyramid](https://arxiv.org/abs/1602.06359) | TF2 | Char | 69.0% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/joint/main/pyramid.ipynb) | Match Pyramid + Joint Training | TF2 | Char | 70.3% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/ant/main/bert.ipynb) | [BERT](https://arxiv.org/abs/1810.04805) | TF2 | Char | 73.8% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/ant/main/bert_further_pretrain_finetune.ipynb) | BERT + [TAPT](https://arxiv.org/abs/2004.10964) ([\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/ant/main/bert_further_pretrain.ipynb)) | TF2 | Char | 74.3% |

* Joint training

	* set data_1 = 微众银行智能客服

	* set data_2 = 蚂蚁金融语义相似度

	* total size = 100000 + 34334 = 134334

	| BERT | only train by data_1 | only train by data_2 | joint train by data_1 and data_2 |
	| --- | --- | --- | --- |
	| Code | [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/main/bert_finetune.ipynb) | [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/ant/main/bert.ipynb) | [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/joint/main/bert_further_pretrain_finetune.ipynb) |
	| data_1 accuracy | 83.8% | - | 84.2% |
	| data_2 accuracy | - | 73.8% | 74.4% |

---

## Intent Detection and Slot Filling

<img src="https://yuanxiaosc.github.io/2019/03/18/%E6%A7%BD%E5%A1%AB%E5%85%85%E5%92%8C%E6%84%8F%E5%9B%BE%E8%AF%86%E5%88%AB%E4%BB%BB%E5%8A%A1%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5/1.png" height="300">

```
└── finch/tensorflow2/spoken_language_understanding/atis
	│
	├── data
	│   └── glove.840B.300d.txt           # pretrained embedding, download and put here
	│   └── make_data.ipynb               # step 1. run this to generate vocab: word.txt, intent.txt, slot.txt 
	│   └── atis.train.w-intent.iob       # incomplete sample, format <text, slot, intent>
	│   └── atis.test.w-intent.iob        # incomplete sample, format <text, slot, intent>
	│
	├── vocab
	│   └── word.txt                      # list of words in vocabulary
	│   └── intent.txt                    # list of intents in vocabulary
	│   └── slot.txt                      # list of slots in vocabulary
	│	
	└── main              
		└── bigru_clr.ipynb               # step 2. train and evaluate model
		└── ...
```

* Task: [ATIS](https://github.com/yvchen/JointSLU/tree/master/data)（English Data） 

        Training Data: 4978, Testing Data: 893

	* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/data/atis.train.w-intent.iob)

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/vocab/word.txt)

	| Code | Model | Helper | Env | Intent Accuracy | Slot Micro-F1 |
	| --- | --- | --- | --- | --- | --- |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/python/atis/main/crfsuite.ipynb) | [CRF](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf) | - | [crfsuite](https://github.com/scrapinghub/python-crfsuite) | - | 92.6% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/bigru_clr.ipynb) | [Bi-GRU](https://www.ijcai.org/Proceedings/16/Papers/425.pdf) | - | TF2 | 97.4% | 95.4% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/bigru_clr_crf.ipynb) | [Bi-GRU](https://www.ijcai.org/Proceedings/16/Papers/425.pdf) | + CRF | TF2 | 97.2% | 95.8% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/transformer.ipynb) | [Transformer](https://arxiv.org/abs/1706.03762) | - | TF2 | 96.5% | 95.5% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/transformer_time_weight.ipynb)  | Transformer | + [Time Weighting](https://github.com/BlinkDL/minGPT-tuned) | TF2 | 97.2% | 95.6% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/transformer_time_mixing.ipynb)  | Transformer | + [Time Mixing](https://github.com/BlinkDL/minGPT-tuned) | TF2 | 97.5% | 95.8% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/spoken_language_understanding/atis/main/elmo_o1_bigru.ipynb)  | Bi-GRU | + [ELMO](https://arxiv.org/abs/1802.05365) | TF1 | 97.5% | 96.1% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/spoken_language_understanding/atis/main/elmo_o1_bigru_crf.ipynb)   | Bi-GRU | + ELMO + CRF | TF1 | 97.3% | 96.3% |

---

## Retrieval Dialog

* Task: Build a chatbot answering fundamental questions

	* Engine: Elasticsearch

		* [\<Notebook> Sparse Retrieval (split by char)](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/es/free_chat/main/default_retrieve.ipynb)

		* [\<Notebook> Sparse Retrieval (split by word)](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/es/free_chat/main/default_retrieve_seg.ipynb)

			Case Analysis

			| 问题 / 拆解方式 | split by char | split by word |
			| --- | --- | --- |
			| Q: 热死了 | 热死了 -> 热 &nbsp; / &nbsp; 死 &nbsp; / &nbsp; 了 | 热死了 -> 热 &nbsp; / &nbsp; 死了 |

			| 问题 / 模型回复 | split by char | split by word |
			| --- | --- | --- |
			| Q: 热死了 | Q: 热死了 -> Q: 想死你了 | Q: 热死了 -> Q: 热 |
			
			split by word is more robust
			
		* [\<Notebook> Dense Retrieval](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/es/free_chat/main/dense_retrieve.ipynb)

		   Case Analysis

			| 问题 / 模型回复 | Sparse Retrieval | Dense Retrieval |
			| --- | --- | --- |
			| Q: 我喜欢你 | Q: 我喜欢你 -> Q: 我喜欢看书 | Q: 我喜欢你 -> Q: 我爱你 |
			
			dense retrieval is more robust

		* [\<Notebook> Dense Retrieval (Bert)](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/es/free_chat/main/dense_retrieve_bert_hub.ipynb)

		* [\<Notebook> Dense Retrieval (Cross-lingual)](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/es/free_chat/main/dense_retrieve_cross_lingual.ipynb)

---

## Semantic Parsing

<img src="https://pic3.zhimg.com/v2-fa2cdccee8c725af42564b37741ba47a_b.jpg">

```
└── finch/tensorflow2/semantic_parsing/tree_slu
	│
	├── data
	│   └── glove.840B.300d.txt     	# pretrained embedding, download and put here
	│   └── make_data.ipynb           	# step 1. run this to generate vocab: word.txt, intent.txt, slot.txt 
	│   └── train.tsv   		  	# incomplete sample, format <text, tokenized_text, tree>
	│   └── test.tsv    		  	# incomplete sample, format <text, tokenized_text, tree>
	│
	├── vocab
	│   └── source.txt                	# list of words in vocabulary for source (of seq2seq)
	│   └── target.txt                	# list of words in vocabulary for target (of seq2seq)
	│	
	└── main
		└── lstm_seq2seq_tf_addons.ipynb           # step 2. train and evaluate model
		└── ......
		
```

* Task: [Semantic Parsing for Task Oriented Dialog](https://arxiv.org/abs/1810.07942)（English Data）

        Training Data: 31279, Testing Data: 9042

	* [\<Text File>: Data Example](https://github.com/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/data/train.tsv)

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/data/make_data.ipynb)
	
	* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/vocab/target.txt)

	| Code | Reference | Env | Testing Exact Match |
	| --- | --- | --- | --- |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/gru_seq2seq_tf_addons_clr.ipynb) | [GRU Seq2Seq](https://arxiv.org/abs/1606.01933) | TF2 | 74.1% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/lstm_seq2seq_tf_addons_clr.ipynb) | LSTM Seq2Seq | TF2 | 74.1% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/gru_pointer_tf_addons_clr.ipynb) | [GRU Pointer-Generator](https://arxiv.org/abs/1704.04368) | TF2 | 80.4% |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/gru_pointer_tf_addons_clr_char.ipynb) | GRU Pointer-Generator + Char Embedding | TF2 | 80.7% |

	The Exact Match result is higher than [original paper](https://arxiv.org/abs/1810.07942)

---

## Knowledge Graph Completion

```
└── finch/tensorflow2/knowledge_graph_completion/wn18
	│
	├── data
	│   └── download_data.ipynb       	# step 1. run this to download wn18 dataset
	│   └── make_data.ipynb           	# step 2. run this to generate vocabulary: entity.txt, relation.txt
	│   └── wn18  		          	# wn18 folder (will be auto created by download_data.ipynb)
	│   	└── train.txt  		  	# incomplete sample, format <entity1, relation, entity2> separated by \t
	│   	└── valid.txt  		  	# incomplete sample, format <entity1, relation, entity2> separated by \t 
	│   	└── test.txt   		  	# incomplete sample, format <entity1, relation, entity2> separated by \t
	│
	├── vocab
	│   └── entity.txt                  	# incomplete sample, list of entities in vocabulary
	│   └── relation.txt                	# incomplete sample, list of relations in vocabulary
	│	
	└── main              
		└── distmult_1-N.ipynb    	# step 3. train and evaluate model
		└── ...
```

* Task: WN18

        Training Data: 141442, Testing Data: 5000

	* [\<Notebook>: Download Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/download_data.ipynb)
	
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/wn18/train.txt)
	
	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/vocab/relation.txt)
	
	* We use the idea of [multi-label classification](https://arxiv.org/abs/1707.01476) to accelerate evaluation

		 <img src="https://pic4.zhimg.com/80/v2-8cd8481856f101af45501078b04456bb_720w.jpg">

	| Code | Reference | Env | MRR | Hits@10 | Hits@3 | Hits@1 |
	| --- | --- | --- | --- | --- | --- | --- |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/distmult_1-N_clr.ipynb) | [DistMult](https://arxiv.org/abs/1412.6575) | TF2 | 0.797 | 0.938 | 0.902 | 0.688 |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/tucker_1-N_clr.ipynb) | [TuckER](https://arxiv.org/abs/1901.09590) | TF2 | 0.885 | 0.939 | 0.909 | 0.853 |
	| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/complex_1-N_clr.ipynb) | [ComplEx](https://arxiv.org/abs/1606.06357) | TF2 | 0.938 | 0.958 | 0.948 | 0.925 |

---

## Knowledge Graph Tools

* Data Scraping

	* [Using Scrapy](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/scrapy/car.ipynb)

	* [Downloaded](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/scrapy/car.csv)

* SPARQL

	* [WN18 Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/rdf_sparql_test.ipynb)

* Neo4j + Cypher

	* [Getting Started](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/neo4j/install_neo4j.ipynb)

---

## Knowledge Base Question Answering

<img src="https://upload-images.jianshu.io/upload_images/17747892-e994edc3518b2d58.png?imageMogr2/auto-orient/strip|imageView2/2/w/880" height="350">

* Rule-based System（基于规则的系统）
	
	For example, we want to answer the following questions with car knowledge:
	
	```
		What is BMW?
        	I want to know about the BMW
        	Please introduce the BMW to me
        	How is the BMW?
        	How is the BMW compared to the Benz?
	```

	* [\<Notebook> Regular Expression](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/python/kbqa/regex.ipynb)

	* [\<Notebook> Regular Expression + POS Feature](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/python/kbqa/rule_based_qa.ipynb)

---

## Multi-hop Question Answering

<img src="https://github.com/DSKSD/DeepNLP-models-Pytorch/blob/master/images/10.dmn-architecture.png" width='500'>

```
└── finch/tensorflow1/question_answering/babi
	│
	├── data
	│   └── make_data.ipynb           		# step 1. run this to generate vocabulary: word.txt 
	│   └── qa5_three-arg-relations_train.txt       # one complete example of babi dataset
	│   └── qa5_three-arg-relations_test.txt	# one complete example of babi dataset
	│
	├── vocab
	│   └── word.txt                  		# complete list of words in vocabulary
	│	
	└── main              
		└── dmn_train.ipynb
		└── dmn_serve.ipynb
		└── attn_gru_cell.py
```

* Task: [bAbI](https://research.fb.com/downloads/babi/)（English Data）

	* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/data/qa5_three-arg-relations_test.txt)
	
	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/data/make_data.ipynb)
	
	* Model: [Dynamic Memory Network](https://arxiv.org/abs/1603.01417)
	
		* TensorFlow 1
		
			* [\<Notebook> DMN -> 99.4% Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/main/dmn_train.ipynb)
			
			* [Inference](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/main/dmn_serve.ipynb)

---

## Text Processing Tools

* Word Extraction

	* Chinese

		* [\<Notebook>: Regex Expression Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/python/regex/zhcn_extract.ipynb)

* Word Segmentation

	* Chinese
	
		* [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/custom_op/tf_jieba.ipynb) Jieba TensorFlow Op purposed by [Junwen Chen](https://github.com/applenob)

* Topic Modelling

	* Data: [2373 Lines of Book Titles](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/spark/topic_modelling/book_titles/all_book_titles.txt)（English Data）

		* Model: TF-IDF + LDA

			* Sklearn + pyLDAvis
			
				* [\<Notebook> TF + IDF + LDA](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/topic_modelling/book_titles/lda.ipynb)
				
					[PySpark](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/spark/topic_modelling/book_titles/lda.ipynb) implementation here
				
				* [\<Notebook> Visualization](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/topic_modelling/book_titles/lda.html#topic=1&lambda=1&term=)

---

## Recommender System

<img src="https://github.com/PaddlePaddle/book/blob/develop/05.recommender_system/image/rec_regression_network.png" width='500'>

```
└── finch/tensorflow1/recommender/movielens
	│
	├── data
	│   └── make_data.ipynb           		# run this to generate vocabulary
	│
	├── vocab
	│   └── user_job.txt
	│   └── user_id.txt
	│   └── user_gender.txt
	│   └── user_age.txt
	│   └── movie_types.txt
	│   └── movie_title.txt
	│   └── movie_id.txt
	│	
	└── main              
		└── dnn_softmax.ipynb
		└── ......
```

* Task: [Movielens 1M](https://grouplens.org/datasets/movielens/1m/)（English Data）
	
        Training Data: 900228, Testing Data: 99981, Users: 6000, Movies: 4000, Rating: 1-5

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/data/make_data.ipynb)

		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/data/train.txt)

	* Model: [Fusion](https://www.paddlepaddle.org.cn/documentation/docs/en/1.5/beginners_guide/basics/recommender_system/index_en.html)
	
		| Code | Scoring | LR Decay | Env | Testing MAE |
		| --- | --- | --- | --- | --- |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_sigmoid.ipynb) | Sigmoid (Continuous) | Exponential | TF1 | 0.663 |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_sigmoid_clr.ipynb) | Sigmoid (Continuous) | Cyclical | TF1 | 0.661 |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_softmax.ipynb) | Softmax (Discrete) | Exponential | TF1 | 0.633 |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_softmax_clr.ipynb) | Softmax (Discrete) | Cyclical | TF1 | 0.628 |
		
		The MAE results seem better than the [all the results here](http://mymedialite.net/examples/datasets.html) and [all the results here](https://test.pypi.org/project/scikit-surprise/)

---

## Multi-turn Dialogue Rewriting

<img src="https://pic1.zhimg.com/80/v2-d80efd57b81c6ece955a247ca7247db4_1440w.jpg" width="600">

```
└── finch/tensorflow1/multi_turn_rewrite/chinese/
	│
	├── data
	│   └── make_data.ipynb         # run this to generate vocab, split train & test data, make pretrained embedding
	│   └── corpus.txt		# original data downloaded from external
	│   └── train_pos.txt		# processed positive training data after {make_data.ipynb}
	│   └── train_neg.txt		# processed negative training data after {make_data.ipynb}
	│   └── test_pos.txt		# processed positive testing data after {make_data.ipynb}
	│   └── test_neg.txt		# processed negative testing data after {make_data.ipynb}
	│
	├── vocab
	│   └── cc.zh.300.vec		# fastText pretrained embedding downloaded from external
	│   └── char.npy		# chinese characters and their embedding values (300 dim)	
	│   └── char.txt		# list of chinese characters used in this project 
	│	
	└── main              
		└── baseline_lstm_train.ipynb
		└── baseline_lstm_predict.ipynb
		└── ...
```

* Task: 20k 腾讯 AI 研发数据（Chinese Data）

	```
	data split as: training data (positive): 18986, testing data (positive): 1008

	Training data = 2 * 18986 because of 1:1 Negative Sampling
	```

	* [\<Text File>: Full Data](https://github.com/chin-gyou/dialogue-utterance-rewriter/blob/master/corpus.txt)
	
	* [\<Notebook>: Make Data & Vocabulary & Pretrained Embedding](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/data/make_data.ipynb)

			There are six incorrect data and we have deleted them

		* [\<Text File>: Positive Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/data/train_pos.txt)
		
		* [\<Text File>: Negative Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/data/train_neg.txt)

	* Model (results can be compared to [here](https://github.com/liu-nlper/dialogue-utterance-rewriter) with the same dataset)

		| Code | Model | Env | Exact Match | BLEU-1 | BLEU-2 | BLEU-4 |
		| --- | --- | --- | --- | --- | --- | --- |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_lstm_train_clr.ipynb) | LSTM Seq2Seq + [Dynamic Memory](https://arxiv.org/abs/1603.01417) | TF1 | 56.2% | 94.6 | 89.1 | 78.5 |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_gru_train_clr_multi_attn.ipynb) | GRU Seq2Seq + Dynamic Memory | TF1 | 56.2% | 95.0 | 89.5 | 78.9 |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/pointer_gru_train_clr.ipynb) | GRU [Pointer](https://arxiv.org/abs/1506.03134) | TF1 | 59.2% | 93.2 | 87.7 | 77.2 |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/pointer_gru_train_clr_multi_attn_.ipynb) | GRU [Pointer](https://arxiv.org/abs/1506.03134) + Multi-Attention | TF1 | 60.2% | 94.2 | 88.7 | 78.3 |

	* Deployment: [first export the model](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_lstm_export.ipynb)

		| Inference Code | Environment |
		| --- | --- |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_lstm_predict.ipynb) | Python |
		| [\<Notebook>](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/java/MultiDialogInference/src/ModelInference.java) | Java |

	* Despite End-to-End, this problem can also be decomposed into two stages

		* **Stage 1 (Fast). Detecting the (missing or referred) keywords from the context**
		
			which is a sequence tagging task with sequential complexity ```O(1)```

		* Stage 2 (Slow). Recombine the keywords with the query based on language fluency
			
			which is a sequence generation task with sequential complexity ```O(N)```

			```
			For example, for a given query: "买不起" and the context: "成都房价是多少 不买就后悔了成都房价还有上涨空间"

			First retrieve the keyword "成都房" from the context which is very important

			Then recombine the keyword "成都房" with the query "买不起" which becomes "买不起成都房"
			```
		
		* For Stage 1 (sequence tagging for retrieving the keywords), the experiment results are:

			| Code | Model | Env | Recall | Precision | Exact Match |
			| --- | --- | --- | --- | --- | --- |
			| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese_tagging/main/tagging_only_pos.ipynb) | Bi-GRU| TF1 | 79.6% | 78.7% | 42.6% 
			| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/multi_turn_rewrite/chinese_tagging/main/bert_finetune.ipynb) | BERT | TF2 | 93.6% | 83.1% | 71.6% |

	* However, there is still a practical problem to prefine whether the query needs to be rewritten or not

		* if not, we just simply skip the rewriter and pass the query to the next stage

		* there are actually three situations needs to be classified

			* 0: the query does not need to be rewritten because it is irrelevant to the context

				```
				你喜欢五月天吗	超级喜欢阿信	中午出去吃饭吗
				```

			* 1: the query needs to be rewritten

				```
				你喜欢五月天吗	超级喜欢阿信	你喜欢他的那首歌 -> 你喜欢阿信的那首歌
				```

			* 2: the query does not need to be rewritten because it already contains enough information

				```
				你喜欢五月天吗	超级喜欢阿信	你喜欢阿信的那首歌
				```

		* therefore, we aim for training the model to jointly predict:

			* intent: three situations {0, 1, 2} whether the query needs to be rewritten or not

			* keyword extraction: extract the missing or referred keywords in the context

		* [\<Text File>: Positive Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/multi_turn_rewrite/chinese_tagging/data/test_pos_tag.txt)
		
		* [\<Text File>: Negative Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/multi_turn_rewrite/chinese_tagging/data/test_neg_tag.txt)

		* [\<Notebook> BERT (chinese_base)](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/multi_turn_rewrite/chinese_tagging/main/bert_joint_finetune.ipynb)
			
			-> Intent: 97.9% accuracy

			-> Keyword Extraction: 90.2% recall &nbsp; 80.7% precision &nbsp; 64.3% exact match

---

## Generative Dialog

```
└── finch/tensorflow1/free_chat/chinese_lccc
	│
	├── data
	│   └── LCCC-base.json           	# raw data downloaded from external
	│   └── LCCC-base_test.json         # raw data downloaded from external
	│   └── make_data.ipynb           	# step 1. run this to generate vocab {char.txt} and data {train.txt & test.txt}
	│   └── train.txt           		# processed text file generated by {make_data.ipynb}
	│   └── test.txt           			# processed text file generated by {make_data.ipynb}
	│
	├── vocab
	│   └── char.txt                	# list of chars in vocabulary for chinese
	│   └── cc.zh.300.vec			# fastText pretrained embedding downloaded from external
	│   └── char.npy			# chinese characters and their embedding values (300 dim)	
	│	
	└── main
		└── lstm_seq2seq_train.ipynb    # step 2. train and evaluate model
		└── lstm_seq2seq_infer.ipynb    # step 4. model inference
		└── ...
```

* Task: [Large-scale Chinese Conversation Dataset](https://github.com/thu-coai/CDial-GPT)

        Training Data: 5000000 (sampled due to small memory), Testing Data: 19008
	
	* Data
		
		* [\<Text File>: Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/data/train.txt)

		* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/free_chat/chinese_lccc/data/make_data.ipynb)

			* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/free_chat/chinese_lccc/vocab/char.txt)

	* Model

		| Code | Model | Env | Test Case | Perplexity
		| --- | --- | --- | --- | --- |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/transformer_train.ipynb) | Transformer Encoder + LSTM Generator | TF1 | [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/transformer_infer.ipynb) | 42.465 |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_seq2seq_train.ipynb) | LSTM Encoder + LSTM Generator | TF1 | [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_seq2seq_infer.ipynb) | 41.250 |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_pointer_train.ipynb) | LSTM Encoder + LSTM [Pointer-Generator](https://arxiv.org/abs/1704.04368) | TF1 | [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_pointer_infer.ipynb) | 36.525 |

	* If you want to deploy model in Java production

		```
		└── FreeChatInference
			│
			├── data
			│   └── transformer_export/
			│   └── char.txt
			│   └── libtensorflow-1.14.0.jar
			│   └── tensorflow_jni.dll
			│
			└── src              
				└── ModelInference.java
		```

		* [\<Notebook> Java Inference](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/java/FreeChatInference/src/ModelInference.java)

		* If you don't know the input and output node names in Java, you can display the node names:

			```
			!saved_model_cli show --dir ../model/xxx/1587959473/ --tag_set serve --signature_def serving_default
			```

	* Large Pre-trained [GPT](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)
	
		| Test Case | Model | Env | Author |
		| --- | --- | --- | --- |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/gpt_lccc_base.ipynb) |  GPT LCCC base | TF1 + [bert4keras](https://github.com/bojone/bert4keras) | [Tsinghua University](https://github.com/thu-coai/CDial-GPT) |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/gpt_lccc_large.ipynb) | GPT LCCC large | TF1 + [bert4keras](https://github.com/bojone/bert4keras) | [Tsinghua University](https://github.com/thu-coai/CDial-GPT) |
		| [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/nezha_gpt.ipynb) | NEZHA LM | TF1 + [bert4keras](https://github.com/bojone/bert4keras) | [Jianlin Su](https://github.com/bojone/nezha_gpt_dialog) |

	* Diverse Response Comparsion

		| 问题 / 模型回复 | Our RNN | GPT base | GPT large | Nezha |
		| --- | --- | --- | --- | --- |
		| Q: 你好 | A: 你好 | A: 你好 | A: 你好，你好，我是大好人。 | A: 您好 |
		| Q: 早上好 | A: 早上好 | A: 早上好，好久不见 | A: 早上好！ | A: 早啊早啊 |
		| Q: 晚上好 | A: 晚上好 | A: 你好 | A: 好久不见 | A: 晚上好！ |
		| Q: 再见 | A: 再见 | A: 不见 | A: 你是不是不爱我了 | A: 我要走了 |
		| Q: 好久不见 | A: 好久不见 | A: 是呀，好久不见。 | A: 是啊，好久不见。 | A: 是啊，最近怎么样啊 |
		| Q: 想死你了 | A: 我也是 | A: 我也想你， | A: 我不信 | A: 哈哈哈哈哈 |
		| Q: 谢谢你 | A: 不客气 | A: 不用谢 | A: 不用 | A: 不谢不谢， |
		| Q: 爱你 | A: 爱你 | A: 你是爱我 | A: 我也爱你， | A: 我也爱你，你最美 |
		| Q: 你叫什么名字 | A: 你猜 | A: 我也想知道 | A: 叫你 | A: 你是不是想说我的名字太俗了 |
		| Q: 你几岁了 | A: 问女孩子年龄可不是绅士所为哦 ~ | A: 你猜 | A: 你猜 | A: 我不知道 |
		| Q: 现在几点了 | A: 11点 | A: 你说呢？ | A: 十二点 | A: 我也不知道啊 |
		| Q: 今天天气怎么样 | A: 下雨了 | A: 今天很冷，但是我还是觉得很冷 | A: 很好 | A: 不错 |
		| Q: 我们现在在哪里 | A: 在这里 | A: 我在这里等你呢 | A: 我不知道你在说什么 | A: 在我们宿舍楼下。。。 |
		| Q: 你能给我讲个笑话吗 | A: 不能 | A: 讲个笑话 | A: 讲给谁听 | A: 不能 |
		| Q: 你是男孩还是女孩呀 | A: 我是萌妹纸 | A: 你猜 | A: 你猜呀 | A: 我是萌妹纸 |
		| Q: 你会几种语言呀 | A: 你猜 | A: 不会 | A: 你猜呀~ | A: 只会说一句 |
		| Q: 你能陪我玩吗 | A: 可以啊 | A: 可以 | A: 我可以啊 | A: 你猜 |
		| Q: 说话可以大声一点吗 | A: 可以 | A: 可以的 | A: 不要 | A: 我不 |
		| Q: 天气真好 | A: 是的 | A: 是啊！ | A: 天气好是因为有你 | A: 是啊。 |
		| Q: 天气太糟糕了 | A: 是啊 | A: 是啊 | A: 是啊 | A: 是的 |
		| Q: 下雨了 | A: 是的 | A: 你们那边也下雨了 | A: 是呀 | A: 我也是 |
		| Q: 雨好大 | A: 今天下雨了 | A: 是啊 | A: 是的，好大雨 | A: 对呀，我们还穿短袖 |
		| Q: 我讨厌艳阳天 | A: 我也讨厌艳阳天 | A: 我也是。 | A: 我喜欢阴天 | A: 我喜欢 |
		| Q: 好晒啊 | A: 是啊 | A: 是啊 | A: 是啊！！ | A: 是啊，不是很热的 |
		| Q: 今天好冷 | A: 多穿点衣服 | A: 你们都不来 | A: 你那儿也冷了 | A: 是啊。 |
		| Q: 今天好热 | A: 今天热死了 | A: 热死我了 | A: 对啊 | A: 对啊，我都热疯 |
		| Q: 风好大 | A: 是啊 | A: 风太大了。。。 | A: 你不是在上课嘛 | A: 嗯嗯嗯，风好大，冷 |
		| Q: 雾太大了看不清路 | A: 雾太大了 | A: 哈哈，你这是在逗我笑 | A: 我也看不清 | A: 你是不是傻啊 |
		| Q: 打雷了好可怕 | A: 可怕 | A: 我们这边也是，我们这边下雨了，我们这边也没有下雨。 | A: 是啊 | A: 我们这边下雨了 |
		| Q: 下雪了诶 | A: 下雪了？ | A: 是滴！ | A: 嗯嗯嗯！！好大的雪！！ | A: 下了一天了 |
		| Q: 好烦啊 | A: 我也好烦 | A: 我也是 | A: 我好烦 | A: 我也是 |
		| Q: 好开心 | A: 开心就好 | A: 你是不是想我了 | A: 是呀 | A: 哈哈哈，我也是！ |
		| Q: 太激动了 | A: 激动什么 | A: 我也激动 | A: 哈哈，还是不要了 | A: 哈哈 |
		| Q: 我好难过 | A: 怎么了？ | A: 我也是，我们都是 | A: 不要难过，我陪你 | A: 你怎么难过 |
		| Q: 我想哭 | A: 抱抱 | A: 抱紧你 | A: 我不想哭！你不要想太多！ | A: 哭吧哭吧，我陪你哭 |
		| Q: 太好笑了 | A: 笑死我了 | A: 你也笑了？ | A: 我也觉得 | A: 哈哈 |
		| Q: 我好伤心 | A: 怎么了 | A: 你伤心什么 | A: 我也是 | A: 你是不是想笑死我！！下次再给你拍 |
		| Q: 心好痛 | A: 抱抱 | A: 你心痛什么 | A: 心碎一地了。 | A: 怎么了 |
		| Q: 好累啊 | A: 加油 | A: 你不会还是一个人在寝室吧 | A: 好好休息 | A: 累了就睡觉 |
		| Q: 我好疲惫 | A: 我也是 | A: 我也差不多。。 | A: 我好困 | A: 你不是一直都很疲惫么 |
		| Q: 我爱你 | A: 我也爱你 | A: 我也是 | A: 我们都爱你 | A: 爱我就来见面 |
		| Q: 我讨厌你 | A: 我也讨厌你 | A: 你不是一直很喜欢我 | A: 不是我的 | A: 我也讨厌你 |
		| Q: 你真是太棒啦 | A: 谢谢 | A: 你最棒 | A: 你也棒棒哒 | A: 谢谢！ |
		| Q: 你好厉害啊 | A: 哈哈 | A: 是不是 | A: 你也不差呀 | A: 你也可以的 |
		| Q: 吓死我了 | A: 哈哈 | A: 我都不敢出门了 | A: 哈哈哈哈哈！！！ | A: 哈哈哈哈哈笑死我了 |
		| Q: 我想回家 | A: 我也想回家 | A: 我也想！！ | A: 想家的时候回来，想家的时候离开。 | A: 回来吧，家真的好啊！ |
		| Q: 我想爸妈了 | A: 我也想爸妈 | A: 哈哈 | A: 我也想 | A: 想我吗 |
		| Q: 不知道小孩在家有没有听话 | A: 我也不知道 | A: 没有 | A: 听话的话肯定是会听话的。 | A: 我也是听不懂啊 |
		| Q: 想回家撸猫 | A: 我也想回家 | A: 你也想啊？ | A: 我们这也有一个 | A: 回呀回呀 |
