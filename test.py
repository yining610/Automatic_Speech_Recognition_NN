from dataset import AsrDataset

dataloader = AsrDataset(scr_file='./data/clsp.trnscr', feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames")

dataloader.silence_id
len(dataloader.feature[1])