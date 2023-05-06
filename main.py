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
        padded_word_spellings = pad_sequence([torch.tensor(sample[0]) for sample in batch], batch_first=True, padding_value=23)

        # padded_features: (batch_size, max_feature_length)
        # 0-255 quantized 2-character labels, 256 padding token
        padded_features = pad_sequence([torch.tensor(sample[1]) for sample in batch], batch_first=True, padding_value=256)

        # list_of_unpadded_word_spelling_length: (batch_size)
        list_of_unpadded_word_spelling_length = [len(sample[0]) for sample in batch]

        # list_of_unpadded_feature_length: (batch_size)
        list_of_unpadded_feature_length = [len(sample[1]) for sample in batch]

        return padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length

    else: # test dataset batch: list of features
        # padded_features: (batch_size, max_feature_length)
        # 0-255 quantized 2-character labels, 256 padding token
        padded_features = pad_sequence([torch.tensor(sample) for sample in batch], batch_first=True, padding_value=256)

        # list_of_unpadded_feature_length: (batch_size)
        list_of_unpadded_feature_length = [len(sample) for sample in batch]

        return padded_features, list_of_unpadded_feature_length


def train(train_dataloader, model, ctc_loss, optimizer, device):
    # === write your code here ===
    for idx, data in enumerate(train_dataloader):
        padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length = data
        padded_word_spellings = padded_word_spellings.to(device)
        padded_features = padded_features.to(device)

        log_prob = model(padded_features)

        log_prob = log_prob.transpose(0, 1)
        # loss: (batch_size)
        loss = ctc_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss

def validate(validate_dataloader, model, CTC_loss, device):
    with torch.no_grad():
        for i, (padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length) in enumerate(validate_dataloader):
            padded_word_spellings = padded_word_spellings.to(device)
            padded_features = padded_features.to(device)

            log_prob = model(padded_features)

            log_prob = log_prob.transpose(0, 1)
            # loss: (batch_size)
            loss = CTC_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)

            decoded_letter = decode(log_prob, validate_dataloader.dataset.dataset)
        
    return loss


def decode(log_prob, dataset):
    # === write your code here ===
    # pick the most likely letter for every frame

    picked_ids = torch.argmax(log_prob, dim=2)
    picked_ids = picked_ids.cpu().numpy()
    # convert ids to letters: (batch, seq_len)
    picked_letters = [[dataset.id2letter[id] for id in ids] for ids in picked_ids]

    return picked_letters


def compute_accuracy():
    # === write your code here ===
    pass

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
    model = LSTM_ASR(feature_type="discrete", input_size=64, hidden_size=256, num_layers=2, output_size=len(training_set.dataset.letter2id))
    model.to(device)
    
    # training_set here is Subset object, so we need to access its dataset attribute
    loss_function = torch.nn.CTCLoss(blank=training_set.dataset.blank_id, zero_infinity=True)

    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Training
    num_epochs = 10
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
