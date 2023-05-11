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
    count_beam = 0
    count_greedy = 0
    count_mrd = 0
    for idx, data in enumerate(train_dataloader):
        padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length = data
        padded_features = padded_features.to(device)

        log_prob = model(padded_features)
        words_beam,  words_log_prob_beam = beam_search_decoder(log_prob, train_dataloader.dataset.dataset, k=3)
        words_greedy, words_log_prob_greedy = greedy_search_decoder(log_prob, train_dataloader.dataset.dataset)
        words_mrd, words_ctcloss_mrd = minimum_risk_decoder(log_prob, list_of_unpadded_feature_length, train_dataloader.dataset.dataset, ctc_loss)

        original_words, count_batch_beam =  compute_accuracy(words_beam, padded_word_spellings, train_dataloader.dataset.dataset)
        original_words, count_batch_greedy = compute_accuracy(words_greedy, padded_word_spellings, train_dataloader.dataset.dataset)
        
        count_beam += count_batch_beam
        count_greedy += count_batch_greedy

        log_prob = log_prob.transpose(0, 1)
        # loss: (batch_size)
        loss = ctc_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    accuracy_beam = count_beam / len(train_dataloader.dataset)
    accuracy_greedy = count_greedy / len(train_dataloader.dataset)

    return loss, accuracy_beam, accuracy_greedy

def validate(validate_dataloader, model, CTC_loss, device):
    count_beam = 0
    count_greedy = 0
    count_mrd = 0

    with torch.no_grad():
        for idx, (padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length) in enumerate(validate_dataloader):
            padded_features = padded_features.to(device)

            log_prob = model(padded_features)

            words_beam,  words_log_prob_beam = beam_search_decoder(log_prob, validate_dataloader.dataset.dataset, k=3)

            words_greedy, words_log_prob_greedy = greedy_search_decoder(log_prob, validate_dataloader.dataset.dataset)

            words_mrd, words_ctcloss_mrd = minimum_risk_decoder(log_prob, list_of_unpadded_feature_length, validate_dataloader.dataset.dataset, CTC_loss)

            origin_words, count_batch_beam =  compute_accuracy(words_beam, padded_word_spellings, validate_dataloader.dataset.dataset)
            origin_words, count_batch_greedy = compute_accuracy(words_greedy, padded_word_spellings, validate_dataloader.dataset.dataset)
            origin_words, count_batch_mrd = compute_accuracy(words_mrd, padded_word_spellings, validate_dataloader.dataset.dataset)

            print(words_mrd)
            print(origin_words)
            
            count_beam += count_batch_beam
            count_greedy += count_batch_greedy
            count_mrd += count_batch_mrd

            log_prob = log_prob.transpose(0, 1)
            # loss: (batch_size)
            loss = CTC_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
    
    accuracy_beam = count_beam / len(validate_dataloader.dataset)
    accuracy_greedy = count_greedy / len(validate_dataloader.dataset)
    accuracy_mrd = count_mrd / len(validate_dataloader.dataset)

    return loss, accuracy_beam, accuracy_greedy, accuracy_mrd


def greedy_search_decoder(log_post, dataset):
    """Greedy Search Decoder

    Parameters:
        log_post(Tensor) - the log posterior of network.
        dataset(dataset) - dataset object.
    Outputs:
        words(list) - a list of word spelling.
        max_log_prob(Tensor) - a list of log likelihood of words.

    """

    # find the index of the maximum log posterior and its value
    max_log_prob_per_idx, idx = torch.max(log_post, dim=2)
    max_log_prob = torch.sum(max_log_prob_per_idx, dim=1)

    idx = idx.cpu()

    # compress repeated indices
    compressed_indices = [None] * idx.shape[0]
    for i in range(idx.shape[0]):
        compressed_indices[i] = torch.unique_consecutive(idx[i]).numpy().tolist()
    
    # remove special tokens
    for i, word in enumerate(compressed_indices):
        compressed_indices[i] = [id for id in word if id not in [dataset.blank_id, dataset.silence_id, dataset.pad_id]]
    
    # convert indices to word spelling
    words_spelling = [[dataset.id2letter[id] for id in word] for word in compressed_indices]
    words = [''.join(word) for word in words_spelling]

    return words, max_log_prob


def beam_search_decoder(log_post, dataset, k=3):
    """Beam Search Decoder

    Parameters:
        log_post(Tensor) - the log posterior of network.
        k(int) - beam size of decoder.
        dataset(dataset) - dataset object.
    Outputs:
        indices(Tensor) - a beam of index sequence.
        log_prob(Tensor) - a beam of log likelihood of words.
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
    assert torch.sum(check_best_log_prob - best_log_prob) < 3e-3

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

def minimum_risk_decoder(log_post, list_of_unpadded_feature_length, dataset, ctc_loss):

    # ctc_loss_list: (batch_size, vacab_size)
    ctc_loss_list = torch.zeros(log_post.shape[0], len(dataset.script))
    for i, (log_prob, unpadded_feature_length) in enumerate(zip(log_post, list_of_unpadded_feature_length)):
        for j, word_spelling in enumerate(dataset.script):
            padded_target = pad_sequence(torch.tensor(word_spelling), padding_value=dataset.pad_id).long()
            unpadded_target_length = torch.tensor(len(word_spelling))
            loss = ctc_loss(log_prob, padded_target, unpadded_feature_length, unpadded_target_length)
            print(log_prob.shape)
            print(padded_target.shape)
            print(unpadded_feature_length.shape)
            print(unpadded_target_length.shape)

            ctc_loss_list[i, j] = loss
    
    # find the index of the minimum ctc los
    min_ctcloss, min_index = torch.min(ctc_loss_list, dim=1)
    # gather the corresponding word from dataset.script

    words = [dataset.script[i] for i in min_index]     

    return words, min_ctcloss

   
def compute_accuracy(words, padded_word_spellings, dataset):
    # === write your code here ===
    # recover the word spelling from padded_word_spellings
    # remove special tokens

    padded_word_spellings = padded_word_spellings.numpy().tolist()
    for i, word in enumerate(padded_word_spellings):
        padded_word_spellings[i] = [id for id in word if id not in [dataset.pad_id, dataset.blank_id, dataset.silence_id]]
    
    # convert indices to word spelling
    words_spelling = [[dataset.id2letter[id] for id in word] for word in padded_word_spellings]
    origin_words = [''.join(word) for word in words_spelling]

    count = 0
    for i in range(len(words)):
        if words[i] == origin_words[i]:
            count += 1
    
    return origin_words, count
    

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
        train_loss, accuracy_beam_train, accuracy_greedy_train = train(train_dataloader, model, loss_function, optimizer, device)
        model.eval()
        val_loss, accuracy_beam_val, accuracy_greedy_val = validate(validate_dataloader, model, loss_function, device)
        tqdm.write(f"Epoch: {epoch}, Training Loss: {train_loss}, Training Beam Search Accuracy: {accuracy_beam_train}, Training Greedy Search Accuracy: {accuracy_greedy_train} Validation Loss: {val_loss} Validation Beam Search Accuracy: {accuracy_beam_val}, Validation Greedy Search Accuracy: {accuracy_greedy_val}")
        

    # # Testing (totally by yourself)
    # decode()

    # # Evaluate (totally by yourself)
    # compute_accuracy()


if __name__ == "__main__":
    main()
