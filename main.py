#!/usr/bin/env python

# 2022 Dongji Gao
# 2022 Yiwen Shao

import os
import sys
import string
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataset import AsrDataset
from model import LSTM_ASR
from tqdm import tqdm
import numpy as np
from ctcdecode import CTCBeamDecoder

def collate_fn(batch):
    """
    This function will be passed to your dataloader.
    It pads word_spelling (and features) in the same batch to have equal length.with 0.
    :param batch: batch of input samples
    :return: (required) padded_word_spellings, 
                           padded_features,
                           list_of_unpadded_word_spelling_length (for CTCLoss),
                           list_of_unpadded_feature_length (for CTCLoss)
    """
    # === write your code here ===

    # training datasest batch: list of tuples (word_spelling, features)
    if len(batch[0]) == 2:
        # silence-padded_word_spellings: (batch_size, max_word_spelling_length)
        # 0-22 letters, 23 silence, 24 blank, 25 padding token
        padded_word_spellings = pad_sequence([torch.tensor(sample[0]) for sample in batch], batch_first=True, padding_value=25).long()
        # padded_features: (batch_size, max_feature_length)
        # 0-255 quantized 2-character labels, 256 padding token
        padded_features = pad_sequence([torch.tensor(sample[1]) for sample in batch], batch_first=True, padding_value=256).long()

        # list_of_unpadded_word_spelling_length: (batch_size)
        list_of_unpadded_word_spelling_length = torch.tensor([len(sample[0]) for sample in batch], dtype=torch.long)

        # list_of_unpadded_feature_length: (batch_size)
        list_of_unpadded_feature_length = torch.tensor([len(sample[1]) for sample in batch], dtype=torch.long)

        return padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length

    else: # test dataset batch: list of features
        # padded_features: (batch_size, max_feature_length)
        # 0-255 quantized 2-character labels, 256 padding token
        padded_features = pad_sequence([torch.tensor(sample) for sample in batch], batch_first=True, padding_value=256).long()

        # list_of_unpadded_feature_length: (batch_size)
        list_of_unpadded_feature_length = torch.tensor([len(sample) for sample in batch], dtype=torch.long)

        return padded_features, list_of_unpadded_feature_length


def train(train_dataloader, model, ctc_loss, optimizer, device):
    # === write your code here ===
    for idx, data in enumerate(train_dataloader):
        padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length = data
        padded_features = padded_features.to(device)

        log_prob = model(padded_features)

        # # decode
        # words,  words_log_prob= decode(log_prob, train_dataloader.dataset.dataset, k=3)
        # compute_accuracy(words, padded_word_spellings, train_dataloader.dataset.dataset)

        log_prob = log_prob.transpose(0, 1)
        # loss: (batch_size)
        loss = ctc_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss

def validate(validate_dataloader, model, CTC_loss, device):
    with torch.no_grad():
        for idx, (padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length) in enumerate(validate_dataloader):
            padded_features = padded_features.to(device)

            log_prob = model(padded_features)
            words,  words_log_prob= beam_search_decoder(log_prob, validate_dataloader.dataset.dataset, k=3)
            print(words)
            # compute_accuracy(words, padded_word_spellings, validate_dataloader.dataset.dataset)

            log_prob = log_prob.transpose(0, 1)
            # loss: (batch_size)
            loss = CTC_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)

    return loss


def beam_search_decoder(log_post, dataset, k=3):
    """Beam Search Decoder

    Parameters:

        log_post(Tensor) - the log posterior of network.
        k(int) - beam size of decoder.

    Outputs:

        indices(Tensor) - a beam of index sequence.
        log_prob(Tensor) - a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).
    """
    log_post = log_post.cpu()
    batch_size, seq_length, _ = log_post.shape
    log_prob, indices = log_post[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        # forward update log_prob
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1)
        new_log_prob = torch.zeros(batch_size, k)
        new_indices = torch.zeros((batch_size, k, i+1))

        for idx in range(batch_size):
            log_prob_idx, i = torch.topk(log_prob[idx].flatten(), k, sorted=True)
            top_k_coordinate =  np.array(np.unravel_index(i.numpy(), log_prob[idx].shape)).T
            new_log_prob[idx] = log_prob_idx
            new_indices[idx] = torch.cat((indices[idx][top_k_coordinate[:,-2]], torch.tensor(top_k_coordinate[:,-1]).view(k,-1)), dim = 1)

        log_prob = new_log_prob
        indices = new_indices

    best_indices = indices[:, 0, :] # (batch_size, seq_length)
    best_log_prob = log_prob[:, 0]  # (batch_size)

    # sanity check: best log prob should be the sum of log prob of best indices
    check_best_log_prob = torch.sum(torch.gather(log_post, 2, best_indices.unsqueeze(-1).type(torch.int64)).squeeze(-1), dim=1)
    assert torch.sum(check_best_log_prob - best_log_prob) < 2e-4

    # compress the indices
    compressed_indices = [None] * batch_size
    for idx in range(batch_size):
        compressed_indices[idx] = torch.unique_consecutive(best_indices[idx]).numpy().tolist()
    
    # remove special tokens
    for i, word in enumerate(compressed_indices):
        compressed_indices[i] = [id for id in word if id not in [dataset.blank_id, dataset.silence_id, dataset.pad_id]]

    # convert indices to word spelling
    words_spelling = [[dataset.id2letter[id] for id in word] for word in compressed_indices]
    words = [''.join(word) for word in words_spelling]

    return words, best_log_prob
   
def compute_accuracy(words, padded_word_spellings, dataset):
    # === write your code here ===
    # recover the word spelling from padded_word_spellings
    # remove special tokens
    for i, word in enumerate(padded_word_spellings):
        padded_word_spellings[i] = [id for id in word if id not in [dataset.pad_id, dataset.blank_id, dataset.silence_id]]
    
    # convert indices to word spelling
    print(dataset.id2letter)
    words_spelling = [[dataset.id2letter[id] for id in word] for word in padded_word_spellings]
    origin_words = [''.join(word) for word in words_spelling]

    for i, word in enumerate(origin_words):
        if word == words[i]:
            print("correct")
        else:
            print(words[i])
            print(word)
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_set = AsrDataset(scr_file='./data/clsp.trnscr', feature_type='discrete', feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames")
    test_set = AsrDataset(feature_type='discrete', feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames")

    # split training set into training and validation set
    train_size = int(0.8 * len(training_set))
    validation_size = len(training_set) - train_size
    training_set, validation_set = random_split(training_set, [train_size, validation_size])

    train_dataloader = DataLoader(training_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    validate_dataloader = DataLoader(validation_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # output_size = 25 = 23 letters + silence + blank
    model = LSTM_ASR(feature_type="discrete", input_size=32, hidden_size=256, num_layers=2, output_size=len(training_set.dataset.letter2id))
    model.to(device)
    
    # training_set here is Subset object, so we need to access its dataset attribute
    loss_function = torch.nn.CTCLoss(blank=training_set.dataset.blank_id, reduction='mean', zero_infinity=True)
    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Training
    num_epochs = 50
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = train(train_dataloader, model, loss_function, optimizer, device)
        model.eval()
        val_loss = validate(validate_dataloader, model, loss_function, device)
        tqdm.write(f"Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
        

    # # Testing (totally by yourself)
    # decode()

    # # Evaluate (totally by yourself)
    # compute_accuracy()


if __name__ == "__main__":
    main()
