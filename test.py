from dataset import AsrDataset

training_set = AsrDataset(scr_file='./data/clsp.trnscr', feature_type='discrete', feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames")

test_set = AsrDataset(feature_type='discrete', feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames")

print(training_set.__len__())
