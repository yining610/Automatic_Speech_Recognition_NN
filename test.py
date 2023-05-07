from dataset import AsrDataset
from torch.utils.data import DataLoader, random_split

training_set = AsrDataset(scr_file='./data/clsp.trnscr', feature_type='discrete', feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames")
test_set = AsrDataset(feature_type='discrete', feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames")
# split training set into training and validation set
train_size = int(0.8 * len(training_set))
validation_size = len(training_set) - train_size
training_set, validation_set = random_split(training_set, [train_size, validation_size])

print(training_set.dataset)

import torch
from torch import nn


# Target are to be padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes
# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
print(input.shape)
# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
ctc_loss = nn.CTCLoss()
print(target.shape)
print(input_lengths.shape)
print(target_lengths.shape)
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()