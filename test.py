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
import numpy as np


# Target are to be padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes
k = 3
# Initialize random batch of input vectors, for *size = (T,N,C)
log_post = torch.randn(N, T, C).log_softmax(-1)

batch_size, seq_length, _ = log_post.shape
log_prob, indices = log_post[:, 0, :].topk(k, sorted=True)
indices = indices.unsqueeze(-1)
for i in range(1, seq_length):
    # forward update log_prob
    log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1)
    new_log_prob = torch.zeros(batch_size, k)
    new_indices = torch.zeros((batch_size, k, i+1))

    for idx in range(batch_size):
        log_prob_idx, i = torch.topk(log_prob[idx].flatten(), k)
        top_k_coordinate =  np.array(np.unravel_index(i.numpy(), log_prob[idx].shape)).T
        new_log_prob[idx] = log_prob_idx
        new_indices[idx] = torch.cat((indices[idx][top_k_coordinate[:,-2]], torch.tensor(top_k_coordinate[:,-1]).view(k,-1)), dim = 1)
    
    log_prob = new_log_prob
    indices = new_indices

# check the largest log_prob
best_indices = indices[:, 0, :]
best_log_prob = log_prob[:, 0]
best_log_prob


best_log_prob
check = torch.sum(torch.gather(log_post, 2, best_indices.unsqueeze(-1).type(torch.int64)).squeeze(-1), dim=-1)
print(best_log_prob)
print(torch.sum(torch.gather(log_post, 2, best_indices.unsqueeze(-1).type(torch.int64)).squeeze(-1), dim=-1))

torch.sum(check - best_log_prob)