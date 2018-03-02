from data import bAbI_data_load, DataLoader

import pprint


def bAbI_data_load_test(idx=1):
    data, lens = bAbI_data_load('./temp/qa5_three-arg-relations_train.txt')
    inputs, questions, answers = data
    inputs_len, inputs_sent_len, questions_len, answers_len = lens
    
    print(len(inputs), len(questions), len(answers),
          len(inputs_len), len(inputs_sent_len), len(questions_len), len(answers_len))
    pprint.PrettyPrinter().pprint(inputs[idx])
    print(questions[idx], answers[idx])
    print(inputs_len[idx], inputs_sent_len[idx], questions_len[idx], answers_len[idx])


def vocab_and_len_test():
    dl = DataLoader()
    print(dl.vocab['word2idx'])
    print(dl.params['max_input_len'])
    print(dl.params['max_sent_len'])
    print(dl.params['max_quest_len'])
    print(dl.params['max_answer_len'])


def shape_test():
    dl = DataLoader()
    print(dl.data['val']['inputs'].shape)
    print(dl.data['val']['questions'].shape)
    print(dl.data['val']['answers'].shape)
    print(dl.data['len']['inputs_len'].shape)
    print(dl.data['len']['inputs_sent_len'].shape)
    print(dl.data['len']['questions_len'].shape)
    print(dl.data['len']['answers_len'].shape)


def next_batch_test():
    dl = DataLoader()
    (i, q, a, i_len, i_sent_len, q_len, a_len) = next(dl.next_batch())
    print(i.shape, q.shape, a.shape, i_len.shape, i_sent_len.shape, q_len.shape, a_len.shape)


def main():
    #bAbI_data_load_test()
    #vocab_and_len_test()
    #shape_test()
    #next_batch_test()


if __name__ == '__main__':
    main()