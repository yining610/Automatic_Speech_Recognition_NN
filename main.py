#!/usr/bin/env python

# 2022 Dongji Gao
# 2022 Yiwen Shao

import os
import sys
import string
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataset import AsrDataset
from model import LSTM_ASR
from tqdm import tqdm


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
    
    # padded_word_spellings: (batch_size, max_word_spelling_length)
    padded_word_spellings = pad_sequence([torch.tensor(sample[0]) for sample in batch], batch_first=True, padding_value=-1)

    # padded_features: (batch_size, max_feature_length)
    padded_features = pad_sequence([torch.tensor(sample[1]) for sample in batch], batch_first=True, padding_value=-1)

    # list_of_unpadded_word_spelling_length: (batch_size)
    list_of_unpadded_word_spelling_length = [len(sample[0]) for sample in batch]

    # list_of_unpadded_feature_length: (batch_size)
    list_of_unpadded_feature_length = [len(sample[1]) for sample in batch]

    return padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length

def train(train_dataloader, model, ctc_loss, optimizer, device, epoch, NUM_EPOCHS):
    # === write your code here ===
    loop = tqdm(train_dataloader)
    for idx, data in enumerate(loop):
        padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length = data
        padded_word_spellings = padded_word_spellings.to(device)
        padded_features = padded_features.to(device)
        print(f"Shape of padded_word_spellings: {padded_word_spellings.shape}")

        log_prob = model(padded_features)
        print(f"Shape of log_prob: {log_prob.shape}")

        log_prob = log_prob.transpose(0, 1)
        print(f"Shape of transposed log_prob: {log_prob.shape}")
        
        padded_word_spellings = padded_word_spellings.transpose(0, 1)
        print(f"Shape of transposed padded_word_spellings: {padded_word_spellings.shape}")
        # loss: (batch_size)
        loss = ctc_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())
    
    return loss


def decode():
    # === write your code here ===
    pass

def compute_accuracy():
    # === write your code here ===
    pass

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_set = AsrDataset(scr_file='./data/clsp.trnscr', feature_type='discrete', feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames")
    # test_set = AsrDataset(scr_file='./data/clsp.devscr', feature_type='discrete', feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames")

    train_dataloader = DataLoader(training_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    # test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = LSTM_ASR(feature_type="discrete", input_size=64, hidden_size=256, num_layers=2, output_size=len(training_set.letter2id))
    model.to(device)
    
    # your can simply import ctc_loss from torch.nn
    loss_function = torch.nn.CTCLoss(blank=training_set.blank_id, zero_infinity=True)

    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Training
    num_epochs = 5
    for epoch in range(num_epochs):
        train(train_dataloader, model, loss_function, optimizer, device, epoch, num_epochs)

    # Testing (totally by yourself)
    decode()

    # Evaluate (totally by yourself)
    compute_accuracy()


if __name__ == "__main__":
    main()
