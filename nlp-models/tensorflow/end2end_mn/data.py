from copy import deepcopy
from config import args

import numpy as np
import sklearn


class BaseDataLoader(object):
    def __init__(self):
        self.data = {
            'size': None,
            'val':{
                'inputs': None,
                'questions': None,
                'answers': None,},
            'len':{
                'inputs_len': None,
                'inputs_sent_len': None,
                'questions_len': None,
                'answers_len': None}
        }
        self.vocab = {
            'size': None,
            'word2idx': None,
            'idx2word': None,
        }
        self.params = {
            'vocab_size': None,
            '<start>': None,
            '<end>': None,
            'max_input_len': None,
            'max_sent_len': None,
            'max_quest_len': None,
            'max_answer_len': None,
        }

    def next_batch(self):
        for i in range(0, self.data['size'], args.batch_size):
            yield (self.data['val']['inputs'][i : i + args.batch_size],
                   self.data['val']['questions'][i : i + args.batch_size],
                   self.data['val']['answers'][i : i + args.batch_size],
                   self.data['len']['inputs_len'][i : i + args.batch_size],
                   self.data['len']['inputs_sent_len'][i : i + args.batch_size],
                   self.data['len']['questions_len'][i : i + args.batch_size],
                   self.data['len']['answers_len'][i : i + args.batch_size],)


    def shuffle(self):
        (self.data['val']['inputs'],
         self.data['val']['questions'],
         self.data['val']['answers'],
         self.data['len']['inputs_len'],
         self.data['len']['inputs_sent_len'],
         self.data['len']['questions_len'],
         self.data['len']['answers_len'],) = sklearn.utils.shuffle(self.data['val']['inputs'],
                                                                   self.data['val']['questions'],
                                                                   self.data['val']['answers'],
                                                                   self.data['len']['inputs_len'],
                                                                   self.data['len']['inputs_sent_len'],
                                                                   self.data['len']['questions_len'],
                                                                   self.data['len']['answers_len'],)
        print("Data Shuffled")


class DataLoader(BaseDataLoader):
    def __init__(self, path, is_training=True, vocab=None, params=None):
        super().__init__()
        data, lens = self.load_data(path)
        if is_training:
            self.build_vocab(data)
            self.padding(data, lens)
        else:
            self.demo = data
            self.vocab = vocab
            self.params = params
            self.padding(data, lens, is_training=is_training)


    def load_data(self, path):
        data, lens = bAbI_data_load(path)
        self.data['size'] = len(data[0])
        return data, lens


    def build_vocab(self, data):
        signals = ['<pad>', '<unk>', '<start>', '<end>']
        inputs, questions, answers = data
        i_words = [w for facts in inputs for fact in facts for w in fact if w != '<end>']
        q_words = [w for question in questions for w in question]
        a_words = [w for answer in answers for w in answer if w != '<end>']
        words = list(set(i_words + q_words + a_words))
        self.params['vocab_size'] = len(words) + 4
        self.params['<start>'] = 2
        self.params['<end>'] = 3
        self.vocab['word2idx'] = {word: idx for idx, word in enumerate(signals + words)}
        self.vocab['idx2word'] = {idx: word for word, idx in self.vocab['word2idx'].items()}
        

    def padding(self, data, lens, is_training=True):
        if is_training:
            inputs_len, inputs_sent_len, questions_len, answers_len = lens

            self.params['max_input_len'] = max(inputs_len)
            self.params['max_sent_len'] = max([fact_len for batch in inputs_sent_len for fact_len in batch])
            self.params['max_quest_len'] = max(questions_len)
            self.params['max_answer_len'] = max(answers_len)

            self.data['len']['inputs_len'] = np.array(inputs_len)
            for batch in inputs_sent_len:
                batch += [0] * (self.params['max_input_len'] - len(batch))
            self.data['len']['inputs_sent_len'] = np.array(inputs_sent_len)
            self.data['len']['questions_len'] = np.array(questions_len)
            self.data['len']['answers_len'] = np.array(answers_len)
        else:
            inputs_len, inputs_sent_len, questions_len, answers_len = lens
            self.data['len']['inputs_len'] = np.array(inputs_len)
            for batch in inputs_sent_len:
                batch += [0] * (self.params['max_input_len'] - len(batch))
            self.data['len']['inputs_sent_len'] = np.array(inputs_sent_len)
            self.data['len']['questions_len'] = np.array(questions_len)
            self.data['len']['answers_len'] = np.array(answers_len)

        inputs, questions, answers = deepcopy(data)
        for facts in inputs:
            for sentence in facts:
                for i in range(len(sentence)):
                    sentence[i] = self.vocab['word2idx'].get(sentence[i], self.vocab['word2idx']['<unk>'])
                sentence += [0] * (self.params['max_sent_len'] - len(sentence))
            paddings = [0] * self.params['max_sent_len']
            facts += [paddings] * (self.params['max_input_len'] - len(facts))
        for question in questions:
            for i in range(len(question)):
                question[i] = self.vocab['word2idx'].get(question[i], self.vocab['word2idx']['<unk>'])
            question += [0] * (self.params['max_quest_len'] - len(question))
        for answer in answers:
            for i in range(len(answer)):
                answer[i] = self.vocab['word2idx'].get(answer[i], self.vocab['word2idx']['<unk>'])

        self.data['val']['inputs'] = np.array(inputs)
        self.data['val']['questions'] = np.array(questions)
        self.data['val']['answers'] = np.array(answers)


def bAbI_data_load(path, END=['<end>']):
    inputs = []
    questions = []
    answers = []

    inputs_len = []
    inputs_sent_len = []
    questions_len = []
    answers_len = []

    for d in open(path):
        index = d.split(' ')[0]
        if index == '1':
            fact = []
        if '?' in d:
            temp = d.split('\t')
            q = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']
            a = temp[1].split() + END
            fact_copied = deepcopy(fact)
            inputs.append(fact_copied)
            questions.append(q)
            answers.append(a)

            inputs_len.append(len(fact_copied))
            inputs_sent_len.append([len(s) for s in fact_copied])
            questions_len.append(len(q))
            answers_len.append(len(a))
        else:
            tokens = d.replace('.', '').replace('\n', '').split(' ')[1:] + END
            fact.append(tokens)
    return [inputs, questions, answers], [inputs_len, inputs_sent_len, questions_len, answers_len]
