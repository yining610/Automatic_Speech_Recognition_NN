from dataset import AsrDataset
from torch.utils.data import DataLoader, random_split

training_set = AsrDataset(scr_file='./data/clsp.trnscr', feature_type='discrete', feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames")
test_set = AsrDataset(feature_type='discrete', feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames")
# split training set into training and validation set
train_size = int(0.8 * len(training_set))
validation_size = len(training_set) - train_size
training_set, validation_set = random_split(training_set, [train_size, validation_size])

print(training_set.dataset)