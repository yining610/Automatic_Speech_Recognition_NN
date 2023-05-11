import string
import torch
import librosa
import os
import pandas as pd
from torch.utils.data import Dataset

def read_file_line_by_line(file_name, func=lambda x: x, skip_header=True):
    print("reading file: %s" % file_name)
    res = list()
    with open(file_name, "r") as fin:
        if skip_header:
            fin.readline()  # skip the header
        for line in fin:
            if len(line.strip()) == 0:
                continue
            fields = func(line.strip())
            res.append(fields)
    print("%d lines, done" % len(res))
    return res

class AsrDataset(Dataset):
    def __init__(self, scr_file, dataset_type =  "train", feature_type='discrete', feature_file=None,
                 feature_label_file=None,
                 wav_scp=None, wav_dir=None):
        """
        :param scr_file: clsp.trnscr
        :param feature_type: "quantized" or "mfcc"
        :param feature_file: clsp.trainlbls or clsp.devlbls
        :param feature_label_file: clsp.lblnames
        :param wav_scp: clsp.trnwav or clsp.devwav
        :param wav_dir: wavforms/
        """

        self.feature_type = feature_type
        self.dataset_type = dataset_type
        assert self.feature_type in ['discrete', 'mfcc']
        assert self.dataset_type in ['train', 'test']

        self.blank = "<blank>"
        self.silence = "<sil>"

        # === write your code here ===
        self.scr_file = scr_file
        self.lblnames = read_file_line_by_line(feature_label_file)

        self.lbls = read_file_line_by_line(feature_file, func=lambda x: x.split())

        # 23 letters + silence + blank + pad
        self.pad = "<pad>"

        self.letters = list(string.ascii_lowercase)
        for c in ['k', 'q', 'z']:
            self.letters.remove(c)
        self.silence_id = len(self.letters)
        self.blank_id = len(self.letters) + 1
        self.pad_id = len(self.letters) + 2

        self.letters.append(self.silence)
        self.letters.append(self.blank)
        self.letters.append(self.pad)

        self.letter2id = dict({c: i for i, c in enumerate(self.letters)})
        self.id2letter = dict({i: c for c, i in self.letter2id.items()})
        
        # 256 quantized feature-vector labels
        self.label2id = dict({lbl: i for i, lbl in enumerate(self.lblnames)})
        self.id2label = dict({i: lbl for lbl, i in self.label2id.items()})
        
        if  feature_type == "discrete":
            self.script = read_file_line_by_line(scr_file)
            # convert feature labels to ids
            self.feature = [[self.label2id[lbl] for lbl in line] for line in self.lbls]
            # convert word spelling to ids
            self.script = [[self.letter2id[c] for c in word] for word in self.script
                           ]
        else:
            self.script = [[self.letter2id[c] for c in word] for word in self.script]
            self.feature = self.compute_mfcc(wav_scp, wav_dir)

    def __len__(self):
        """
        :return: num_of_samples
        """
        if self.scr_file is None:
            return len(self.lbls) # number of feature labels in test dataset
        else:
            return len(self.script) # number of word spellings in training dataset

    def __getitem__(self, idx):
        """
        Get one sample each time. Do not forget the leading- and trailing-silence.
        :param idx: index of sample
        :return: spelling_of_word, feature
        """
        # === write your code here ===
        feature = self.feature[idx]
        
        # training dataset
        if self.dataset_type == "train":
            spelling_of_word = self.script[idx]
            # Pad the spelling on each side with a “silence” symbol
            spelling_of_word = [self.silence_id] + spelling_of_word + [self.silence_id]

            return spelling_of_word, feature
        else: # testing dataset
            return feature

    # This function is provided
    def compute_mfcc(self, wav_scp, wav_dir):
        """
        Compute MFCC acoustic features (dim=40) for each wav file.
        :param wav_scp:
        :param wav_dir:
        :return: features: List[np.ndarray, ...]
        """
        features = []
        with open(wav_scp, 'r') as f:
            for wavfile in f:
                wavfile = wavfile.strip()
                if wavfile == 'jhucsp.trnwav':  # skip header
                    continue
                wav, sr = librosa.load(os.path.join(wav_dir, wavfile), sr=None)
                feats = librosa.feature.mfcc(y=wav, sr=16e3, n_mfcc=40, hop_length=160, win_length=400).transpose()
                features.append(feats)
        return features
