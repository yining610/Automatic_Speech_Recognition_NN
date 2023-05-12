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


def train(train_dataloader, model, CTC_loss, optimizer, device):
    # === write your code here ===
    count_beam = 0
    count_greedy = 0
    count_mrd = 0

    dataset = train_dataloader.dataset.dataset
    # set reduction to none to get loss for each sample
    loss_function_mrd = torch.nn.CTCLoss(blank=dataset.blank_id, reduction='none', zero_infinity=True)

    for idx, data in enumerate(train_dataloader):
        padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length = data
        padded_features = padded_features.to(device)

        log_prob = model(padded_features)

        words_beam,  words_log_prob_beam = beam_search_decoder(log_prob, dataset, k=3)
        words_greedy, words_log_prob_greedy = greedy_search_decoder(log_prob, dataset)
        
        words_mrd, words_ctcloss_mrd = minimum_risk_decoder(log_prob, dataset, loss_function_mrd)
        
        origin_words = unpad(padded_word_spellings, dataset)

        count_batch_beam =  compute_accuracy(words_beam, origin_words)
        count_batch_greedy = compute_accuracy(words_greedy, origin_words)
        count_batch_mrd = compute_accuracy(words_mrd, origin_words)

        count_beam += count_batch_beam
        count_greedy += count_batch_greedy
        count_mrd += count_batch_mrd

        log_prob = log_prob.transpose(0, 1)
        # loss: (batch_size)
        loss = CTC_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    accuracy_beam = count_beam / len(train_dataloader.dataset)
    accuracy_greedy = count_greedy / len(train_dataloader.dataset)
    accuracy_mrd = count_mrd / len(train_dataloader.dataset)

    return loss, accuracy_beam, accuracy_greedy, accuracy_mrd

def validate(validate_dataloader, model, CTC_loss, device):
    count_beam = 0
    count_greedy = 0
    count_mrd = 0

    dataset = validate_dataloader.dataset.dataset
    loss_function_mrd = torch.nn.CTCLoss(blank=dataset.blank_id, reduction='none', zero_infinity=True)

    with torch.no_grad():
        for idx, (padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length) in enumerate(validate_dataloader):
            padded_features = padded_features.to(device)

            log_prob = model(padded_features)

            words_beam,  words_log_prob_beam = beam_search_decoder(log_prob, dataset, k=3)
            words_greedy, words_log_prob_greedy = greedy_search_decoder(log_prob, dataset)
            
            words_mrd, words_ctcloss_mrd = minimum_risk_decoder(log_prob, dataset, loss_function_mrd)
            
            origin_words = unpad(padded_word_spellings, dataset)
            print(f"decoded_word: {words_mrd}")
            print(f"origin_word: {origin_words}")
            count_batch_beam =  compute_accuracy(words_beam, origin_words)
            count_batch_greedy = compute_accuracy(words_greedy, origin_words)
            count_batch_mrd = compute_accuracy(words_mrd, origin_words)

            # print(words_mrd)
            # print(origin_words)
            
            count_beam += count_batch_beam
            count_greedy += count_batch_greedy
            count_mrd += count_batch_mrd

            log_prob = log_prob.transpose(0, 1)
            # loss: (batch_size)
            loss = CTC_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
    
    accuracy_beam = count_beam / len(validate_dataloader.dataset)
    accuracy_greedy = count_greedy / len(validate_dataloader.dataset)
    accuracy_mrd = count_mrd / len(validate_dataloader.dataset)
    print(count_mrd)
    print(len(validate_dataloader.dataset))

    return loss, accuracy_beam, accuracy_greedy, accuracy_mrd

def test(test_dataloader, model, device):
    dataset = test_dataloader.dataset 
    loss_function_mrd = torch.nn.CTCLoss(blank=dataset.blank_id, reduction='none', zero_infinity=True)
    for idx, (padded_features, list_of_unpadded_feature_length) in enumerate(test_dataloader):
        padded_features = padded_features.to(device)

        log_prob = model(padded_features)

        words_beam,  words_log_prob_beam = beam_search_decoder(log_prob, dataset, k=3)
        words_greedy, words_log_prob_greedy = greedy_search_decoder(log_prob, dataset)
        words_mrd, words_ctcloss_mrd = minimum_risk_decoder(log_prob, dataset, loss_function_mrd)
        
    return words_beam, words_greedy, words_mrd, words_log_prob_beam, words_log_prob_greedy, words_ctcloss_mrd


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

def minimum_risk_decoder(log_post, dataset, ctc_loss):

    # expand log_post to (batch_size, vocabular_size ,seq_length, output_size)
    log_post = log_post.unsqueeze(1).repeat(1, len(dataset.script), 1, 1)

    words_list = []
    min_ctcloss_list = []

    for batched_log_prob in log_post:
        target = dataset.script
        padded_target = pad_sequence([torch.tensor(sample) for sample in target], batch_first=True, padding_value=dataset.pad_id).long()

        list_of_unpadded_feature_length = [len(feature) for feature in dataset.feature]
        # ensure the length of unpadded feature of all words is not longer than the sequence length of batched log_post
        list_of_unpadded_feature_length = [length if length <= batched_log_prob.shape[1] else batched_log_prob.shape[1] for length in list_of_unpadded_feature_length]
        list_of_unpadded_feature_length = torch.tensor(list_of_unpadded_feature_length, dtype=torch.long)
        list_of_unpadded_target_length = torch.tensor([len(word) for word in target], dtype=torch.long)

        # batched_log_prob: (vocabular_size, seq_length, output_size)
        # padded_target: (vocabular_size, max_word_spelling_length)
        # list_of_unpadded_target_length: (vocabular_size)
        # list_of_unpadded_feature_length: (vocabular_size)
        # loss: (batch_size)
        batched_log_prob = batched_log_prob.transpose(0, 1)
        
        loss = ctc_loss(batched_log_prob, padded_target, list_of_unpadded_feature_length, list_of_unpadded_target_length)

        # find the minimum ctc loss and its index
        min_ctcloss, min_index = torch.min(loss, dim=0)

        word_ids = dataset.script[min_index]
        word = ''.join([dataset.id2letter[id] for id in word_ids])
        words_list.append(word)
        min_ctcloss_list.append(min_ctcloss)
    
    return words_list, min_ctcloss_list


def unpad(padded_word_spellings, dataset):
    # convert padded word spelling to original word
    padded_word_spellings = padded_word_spellings.numpy().tolist()
    for i, word in enumerate(padded_word_spellings):
        padded_word_spellings[i] = [id for id in word if id not in [dataset.pad_id, dataset.blank_id, dataset.silence_id]]
    # convert indices to word spelling
    words_spelling = [[dataset.id2letter[id] for id in word] for word in padded_word_spellings]
    origin_words = [''.join(word) for word in words_spelling]

    return origin_words
   
def compute_accuracy(words, original_words):
    # === write your code here ===

    count = 0
    for i in range(len(words)):
        if words[i] == original_words[i]:
            count += 1
    
    return count
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_set = AsrDataset(scr_file='./data/clsp.trnscr', dataset_type="train", feature_type='discrete', feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames")
    test_set = AsrDataset(scr_file = "./data/clsp.trnscr", dataset_type="test", feature_type='discrete', feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames")

    # split training set into training and validation set
    train_size = int(0.9 * len(training_set))
    validation_size = len(training_set) - train_size
    training_set, validation_set = random_split(training_set, [train_size, validation_size])

    train_dataloader = DataLoader(training_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    validate_dataloader = DataLoader(validation_set, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # output_size = 25 = 23 letters + silence + blank
    model = LSTM_ASR(feature_type="discrete", input_size=32, hidden_size=256, num_layers=1, output_size=len(training_set.dataset.letter2id))
    model.to(device)
    
    # training_set here is Subset object, so we need to access its dataset attribute
    loss_function = torch.nn.CTCLoss(blank=training_set.dataset.blank_id, reduction='mean', zero_infinity=True)
    
    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Training
    num_epochs = 50
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss, accuracy_beam_train, accuracy_greedy_train, accuracy_mrd_train = train(train_dataloader, model, loss_function, optimizer, device)
        model.eval()
        val_loss, accuracy_beam_val, accuracy_greedy_val, accuracy_mrd_val = validate(validate_dataloader, model, loss_function, device)

        tqdm.write(f"Epoch: {epoch}, Training Loss: {train_loss}, Training Beam Search Accuracy: {accuracy_beam_train}, Training Greedy Search Accuracy: {accuracy_greedy_train}, Training Minimum Risk Decode Accuracy: {accuracy_mrd_train}, Validation Loss: {val_loss}, Validation Beam Search Accuracy: {accuracy_beam_val}, Validation Greedy Search Accuracy: {accuracy_greedy_val}, Validation Minimum Risk Decode Accuracy: {accuracy_mrd_val}")
    
    # testing
    model.eval()
    words_beam, words_greedy, words_mrd, words_log_prob_beam, words_log_prob_greedy, words_ctcloss_mrd = test(test_dataloader, model, device)
    print("###################Testing###################")
    print(f"Beam Search Decoded Words: {words_beam}")
    print(f"Beam Search Decoded Forward Log Probability: {words_log_prob_beam}")
    print(f"Greedy Search Decoded Words: {words_greedy}")
    print(f"Greedy Search Decoded Forward Log Probability: {words_log_prob_greedy}")
    print(f"Minimum Risk Decode Words: {words_mrd}")
    print(f"Minimum Risk Decode CTC Loss: {words_ctcloss_mrd}")



if __name__ == "__main__":
    main()
