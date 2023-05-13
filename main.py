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
import argparse
import json
import matplotlib.pyplot as plt

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

    # silence-padded_word_spellings: (batch_size, max_word_spelling_length)
    # 0-22 letters, 23 silence, 24 blank, 25 padding token
    padded_word_spellings = pad_sequence([torch.tensor(sample[0]) for sample in batch], batch_first=True, padding_value=25)
    # padded_features: (batch_size, max_feature_length)
    # 0-255 quantized 2-character labels, 256 padding token
    padded_features = pad_sequence([torch.tensor(sample[1]) for sample in batch], batch_first=True, padding_value=256)
    # list_of_unpadded_word_spelling_length: (batch_size)
    list_of_unpadded_word_spelling_length = [len(sample[0]) for sample in batch]
    # list_of_unpadded_feature_length: (batch_size)
    list_of_unpadded_feature_length = [len(sample[1]) for sample in batch]
    return padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length


def train(train_dataloader, model, CTC_loss, optimizer):
    # === write your code here ===
    dataset = train_dataloader.dataset.dataset
    count_beam = 0
    count_greedy = 0
    # count_mrd = 0
    for idx, data in enumerate(train_dataloader):
        padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length = data
        log_prob = model(padded_features)

        words_beam,  _ = beam_search_decoder(log_prob, dataset, k=3)
        words_greedy, _ = greedy_search_decoder(log_prob, dataset)
        # words_mrd, words_ctcloss_mrd = minimum_risk_decoder(log_prob, dataset, list_of_unpadded_feature_length)

        origin_words = unpad(padded_word_spellings, dataset)
        
        count_batch_beam =  compute_accuracy(words_beam, origin_words)
        count_batch_greedy = compute_accuracy(words_greedy, origin_words)
        # count_batch_mrd = compute_accuracy(words_mrd, origin_words)

        count_beam += count_batch_beam
        count_greedy += count_batch_greedy
        # count_mrd += count_batch_mrd

        log_prob = log_prob.transpose(0, 1)
        # loss: (batch_size)
        loss = CTC_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    accuracy_beam = count_beam / len(train_dataloader.dataset)
    accuracy_greedy = count_greedy / len(train_dataloader.dataset)
    # accuracy_mrd = count_mrd / len(train_dataloader.dataset)

    return loss, accuracy_beam, accuracy_greedy

def validate(validate_dataloader, model, CTC_loss):
    count_beam = 0
    count_greedy = 0
    count_mrd = 0

    dataset = validate_dataloader.dataset.dataset
    with torch.no_grad():
        for idx, (padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length) in enumerate(validate_dataloader):
            log_prob = model(padded_features)
            words_beam,  _ = beam_search_decoder(log_prob, dataset, k=3)
            words_greedy, _ = greedy_search_decoder(log_prob, dataset)

            words_mrd, _ = minimum_risk_decoder(log_prob, dataset, list_of_unpadded_feature_length)

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
        
        # print the last batch
        print(f"Validation Greedy Search Decoded Words: {words_greedy}")
        print(f"Validationo Beam Search Decoded Words: {words_beam}")
        print(f"Validation CTC Decoded Words: {words_mrd}")
        print(f"original words: {origin_words}")

        accuracy_beam = count_beam / len(validate_dataloader.dataset)
        accuracy_greedy = count_greedy / len(validate_dataloader.dataset)
        accuracy_mrd = count_mrd / len(validate_dataloader.dataset)

    return loss, accuracy_beam, accuracy_greedy, accuracy_mrd

def test(test_dataloader, model):
    dataset = test_dataloader.dataset
    words_beam_list = []
    words_greedy_list = []
    words_mrd_list = []
    words_log_prob_beam_list = []
    words_log_prob_greedy_list = []
    words_ctcloss_mrd_list = []
    with torch.no_grad():
        for idx, (_, padded_features, _, list_of_unpadded_feature_length) in enumerate(test_dataloader):
            log_prob = model(padded_features)

            words_beam,  words_log_prob_beam = beam_search_decoder(log_prob, dataset, k=3)
            words_greedy, words_log_prob_greedy = greedy_search_decoder(log_prob, dataset)
            words_mrd, words_ctcloss_mrd = minimum_risk_decoder(log_prob, dataset, list_of_unpadded_feature_length)

            words_beam_list.extend(words_beam)
            words_greedy_list.extend(words_greedy)
            words_mrd_list.extend(words_mrd)
            words_log_prob_beam_list.extend(words_log_prob_beam)
            words_log_prob_greedy_list.extend(words_log_prob_greedy)
            words_ctcloss_mrd_list.extend(words_ctcloss_mrd)

    return words_beam_list, words_greedy_list, words_mrd_list, words_log_prob_beam_list, words_log_prob_greedy_list, words_ctcloss_mrd_list


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

    # compress consecutive repeated indices
    compressed_indices = [None] * idx.shape[0]
    for i in range(idx.shape[0]):
        compressed_indices[i] = torch.unique_consecutive(idx[i]).numpy().tolist()
    
    # remove special tokens
    for i, word in enumerate(compressed_indices):
        compressed_indices[i] = [id for id in word if id not in [dataset.blank_id, dataset.silence_id, dataset.pad_id, dataset.space_id]]
    
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
    # check_best_log_prob = torch.sum(torch.gather(log_post, 2, best_indices.unsqueeze(-1).type(torch.int64)).squeeze(-1), dim=1)
    # assert torch.sum(check_best_log_prob - best_log_prob) < 3e-3

    # compress the indices
    compressed_indices = [None] * batch_size
    for idx in range(batch_size):
        compressed_indices[idx] = torch.unique_consecutive(best_indices[idx]).numpy().tolist()
    
    # remove special tokens
    for i, word in enumerate(compressed_indices):
        compressed_indices[i] = [id for id in word if id not in [dataset.blank_id, dataset.silence_id, dataset.pad_id, dataset.space_id]]

    # convert indices to word spelling
    words_spelling = [[dataset.id2letter[id] for id in word] for word in compressed_indices]
    words = [''.join(word) for word in words_spelling]

    return words, best_log_prob

def minimum_risk_decoder(log_post, dataset, list_of_unpadded_feature_length):
    """CTC decode

    Parameters:
        log_post(Tensor): (batch_size, seq_length, output_size)
        dataset(dataset): dataset object
        ctc_loss(torch.nn.CTCLoss): ctc loss function
        list_of_unpadded_feature_length(Tensor): (batch_size)
    
    Outputs:
        words_list(list): a list of word spelling
        min_ctcloss_list(list): a list of minimum ctc loss
    """

    ctc_loss = nn.CTCLoss(blank=dataset.blank_id)
    words_list = []
    min_ctcloss_list = []
    for log_prob, unpadded_feature_length in zip(log_post, list_of_unpadded_feature_length):
        min_ctcloss = float('inf')
        selected_word = None
        for word in dataset.script:
            input = log_prob.unsqueeze(0)
            # input: (seq_length, 1, output_size)
            input = input.transpose(0, 1)

            # Pad the spelling on each side with a “silence” symbol
            spelling_of_word = [dataset.silence_id] + word + [dataset.silence_id]
            spelling_of_word = [item for sublist in [[i, dataset.space_id] for i in spelling_of_word] for item in sublist][:-1]
            # target: (1, seq_length)
            target = torch.tensor([spelling_of_word])
            
            # unpadded_feature_length: (1)
            list_unpadded_feature_length = [unpadded_feature_length]
            # unpadded_target_length: (1)
            list_unpadded_target_length = [len(spelling_of_word)]

            loss = ctc_loss(log_prob, target, list_unpadded_feature_length, list_unpadded_target_length)       
            
            if loss < min_ctcloss:
                min_ctcloss = loss
                selected_word = "".join([dataset.id2letter[id] for id in word])

        words_list.append(selected_word)
        min_ctcloss_list.append(min_ctcloss)
    
    return words_list, min_ctcloss_list


def unpad(padded_word_spellings, dataset):
    # # convert padded word spelling to original word

    origin_words = []
    padded_word_spellings = padded_word_spellings.numpy().tolist()
    for word in padded_word_spellings:
        word = [id for id in word if id not in [dataset.blank_id, dataset.silence_id, dataset.pad_id, dataset.space_id]]
        word = [dataset.id2letter[id] for id in word]
        word = ''.join(word)
        origin_words.append(word)

    return origin_words
   
def compute_accuracy(words, original_words):
    # === write your code here ===

    count = 0
    for word, original_word in zip(words, original_words):
        if word == original_word:
            count += 1
    
    return count
    

def main():

    training_set = AsrDataset(scr_file='./data/clsp.trnscr', feature_type=args.feature_type, feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames",  wav_scp='./data/clsp.trnwav', wav_dir='./data/waveforms/')
    test_set = AsrDataset(scr_file = "./data/clsp.trnscr", feature_type=args.feature_type, feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames",  wav_scp='./data/clsp.devwav', wav_dir='./data/waveforms/')
    
    # split training set into training and validation set
    train_size = int(0.95 * len(training_set))
    validation_size = len(training_set) - train_size

    generator = torch.Generator().manual_seed(42)
    training_set, validation_set = random_split(training_set, [train_size, validation_size], generator=generator)

    train_dataloader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    validate_dataloader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.verbose:
        print(f"Batch Size: {args.batch_size}")
        print(f"Training set size: {len(train_dataloader.dataset)}")
        print(f"Validation set size: {len(validate_dataloader.dataset)}")
        print(f"Test set size: {len(test_dataloader.dataset)}")
        print(f"letters: {training_set.dataset.letters}")
    
    # output_size = 25 = 23 letters + silence + blank
    model = LSTM_ASR(feature_type=args.feature_type, input_size=40, hidden_size=256, num_layers=2, output_size=len(training_set.dataset.letter2id))

    # training_set here is Subset object, so we need to access its dataset attribute
    loss_function = torch.nn.CTCLoss(blank=training_set.dataset.blank_id)
    
    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Training
    num_epochs = args.epoch
    train_loss_list = []
    # train_mrd_acc_list = []
    train_grd_acc_list = []
    train_beam_acc_list = []

    val_loss_list = []
    val_mrd_acc_list = []
    val_grd_acc_list = []
    val_beam_acc_list = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss, accuracy_beam_train, accuracy_greedy_train = train(train_dataloader, model, loss_function, optimizer)
        model.eval()
        val_loss, accuracy_beam_val, accuracy_greedy_val, accuracy_mrd_val = validate(validate_dataloader, model, loss_function)
        
        train_loss_list.append(train_loss.item())
        train_grd_acc_list.append(accuracy_greedy_train)
        train_beam_acc_list.append(accuracy_beam_train)
        # train_mrd_acc_list.append(accuracy_mrd_train)

        val_loss_list.append(val_loss.item())
        val_grd_acc_list.append(accuracy_greedy_val)
        val_beam_acc_list.append(accuracy_beam_val)
        val_mrd_acc_list.append(accuracy_mrd_val)
        
        tqdm.write(f"Epoch: {epoch}, Training Loss: {train_loss}, accuracy_beam_train: {accuracy_beam_train}, accuracy_greedy_train: {accuracy_greedy_train}, Validation Loss: {val_loss}, Validation Beam Search Accuracy: {accuracy_beam_val}, Validation Greedy Search Accuracy: {accuracy_greedy_val}, Validation Minimum Risk Decode Accuracy: {accuracy_mrd_val}")
    
    # plot training and validation loss
    plt.plot(train_loss_list, label="Training Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./figures/{args.feature_type}_loss.png", dpi=300)

    # plot training and validation accuracy
    plt.clf()
    plt.plot(train_grd_acc_list, label="Training Greedy Search Accuracy")
    plt.plot(val_grd_acc_list, label="Validation Greedy Search Accuracy")
    plt.plot(train_beam_acc_list, label="Training Beam Search Accuracy")
    plt.plot(val_beam_acc_list, label="Validation Beam Search Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"./figures/{args.feature_type}_accuracy.png", dpi=300)

    plt.clf()
    plt.plot(val_mrd_acc_list, label="Validation Minimum Risk Decode Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"./figures/{args.feature_type}_mrd_accuracy.png", dpi=300)

    # testing
    model.eval()
    words_beam, words_greedy, words_mrd, words_log_prob_beam, words_log_prob_greedy, words_ctcloss_mrd = test(test_dataloader, model)
   
    words_log_prob_beam = [i.item() for i in words_log_prob_beam]
    words_log_prob_greedy = [i.item() for i in words_log_prob_greedy]
    words_ctcloss_mrd = [i.item() for i in words_ctcloss_mrd]

    # save testing result to json file
    store_files_path = f"./{args.feature_type}_test_result.json"
    with open(store_files_path, 'w') as f:
        data = [None] * len(words_beam)
        for i in range(len(words_beam)):
            data[i] = {"Test Id": i,
                       "Beam Search Decode": words_beam[i], 
                       "Beam Search Forward Log Prob": words_log_prob_beam[i],
                       "Greedy Search Decode": words_greedy[i],
                       "Greedy Search Forward Log Prob": words_log_prob_greedy[i],
                       "Minimum Risk Decode": words_mrd[i],
                       "Minimum Risk CTC Loss": words_ctcloss_mrd[i]}
        json.dump(data, f, indent=4)

    # Save the model 
    torch.save(model.state_dict(), f"./checkpoints/{args.feature_type}_model.pt")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    argparser.add_argument("-e", "--epoch", type=int, default=10, help="number of epochs")
    argparser.add_argument("-f", "--feature_type", type=str, default="discrete", help="feature type: discrete or mfcc")
    argparser.add_argument("-b", "--batch_size", type=int, default=32, help="batch size")

    args = argparser.parse_args()

    main()
