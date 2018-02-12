from scipy.io import wavfile
from python_speech_features import mfcc
from tqdm import tqdm
from utils import sparse_tuple_from

import os
import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        wav_files = [f for f in os.listdir('./data') if f.endswith('.wav')]
        text_files = [f for f in os.listdir('./data') if f.endswith('.txt')]

        inputs, targets = [], []
        for (wav_file, text_file) in tqdm(zip(wav_files, text_files), total=len(wav_files), ncols=70):
            path = './data/' + wav_file
            try:
                fs, audio = wavfile.read(path)
            except:
                continue
            input = mfcc(audio, samplerate=fs, nfft=1024)
            inputs.append(input)
            with open('./data/'+text_file) as f:
                targets.append(f.read())

        self.seq_lens = np.array([len(input) for input in inputs])
        self.inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs)

        chars = list(set([c for target in targets for c in target]))
        self.num_classes = len(chars) + 1

        self.idx2char = {idx: char for idx, char in enumerate(chars)}
        self.char2idx = {char: idx for idx, char in self.idx2char.items()}

        self.targets = [[self.char2idx[c] for c in target] for target in targets]

        self.inputs_val = np.expand_dims(self.inputs[-1], 0)
        self.seq_lens_val = np.atleast_1d(self.seq_lens[-1])
        self.targets_val = self.targets[-1]

        self.inputs = self.inputs[:-1]
        self.seq_lens = self.seq_lens[:-1]
        self.targets = self.targets[:-1]

    def next_batch(self):
        batch_size = self.batch_size
        for i in range(0, len(self.inputs), batch_size):
            yield  (self.inputs[i : i+batch_size],
                    self.seq_lens[i : i+batch_size],
                    sparse_tuple_from(self.targets[i : i+batch_size]))
