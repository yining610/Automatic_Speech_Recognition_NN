import torch
from dataset import AsrDataset
from main import collate_fn, validate, test, beam_search_decoder, greedy_search_decoder, minimum_risk_decoder
from torch.utils.data import DataLoader, random_split
from model import LSTM_ASR


# load model from checkpoints
discrete_model = LSTM_ASR(feature_type="discrete", input_size=40, hidden_size=256, num_layers=2, output_size=27)
discrete_model.load_state_dict(torch.load("./checkpoints/discrete_model.pt"))
mfcc_model = LSTM_ASR(feature_type="mfcc", input_size=40, hidden_size=256, num_layers=2, output_size=27)
mfcc_model.load_state_dict(torch.load("./checkpoints/mfcc_model.pt"))


# load validation data
discrete_training_set = AsrDataset(scr_file='./data/clsp.trnscr', feature_type= "discrete",feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames",  wav_scp='./data/clsp.trnwav', wav_dir='./data/waveforms/')
discrete_test_set = AsrDataset(scr_file = "./data/clsp.trnscr", feature_type="discrete", feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames",  wav_scp='./data/clsp.devwav', wav_dir='./data/waveforms/')

mfcc_training_set = AsrDataset(scr_file='./data/clsp.trnscr', feature_type= "mfcc",feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames",  wav_scp='./data/clsp.trnwav', wav_dir='./data/waveforms/')
mfcc_test_set = AsrDataset(scr_file = "./data/clsp.trnscr", feature_type="mfcc", feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames",  wav_scp='./data/clsp.devwav', wav_dir='./data/waveforms/')

# split training set into training and validation set
train_size = int(0.95 * len(discrete_training_set))
validation_size = len(discrete_training_set) - train_size
generator = torch.Generator().manual_seed(42)

# create validation dataset
discrete_training_set, discrete_validation_set = random_split(discrete_training_set, [train_size, validation_size], generator=generator)
mfcc_training_set, mfcc_validation_set = random_split(mfcc_training_set, [train_size, validation_size], generator=generator)

discrete_val_dataloader = DataLoader(discrete_validation_set, batch_size=16, shuffle=False, collate_fn=collate_fn)
discrete_test_dataloader = DataLoader(discrete_test_set, batch_size=16, shuffle=False, collate_fn=collate_fn)

mfcc_val_dataloader = DataLoader(mfcc_validation_set, batch_size=16, shuffle=False, collate_fn=collate_fn)
mfcc_test_dataloader = DataLoader(mfcc_test_set, batch_size=16, shuffle=False, collate_fn=collate_fn)

# evaluate on validation set
loss_function = torch.nn.CTCLoss(blank=discrete_test_set.blank_id)
print("Evaluating on Validation Set Using Minimum CTC Loss Decoder For Discrete Model")
discrete_val_loss, discrete_accuracy_beam_val, discrete_accuracy_greedy_val, discrete_accuracy_mrd_val = validate(discrete_val_dataloader, discrete_model, loss_function)
print(f"Validation Accuracy: {discrete_accuracy_mrd_val}")

print("Evaluating on Validation Set Using Minimum CTC Loss Decoder For MFCC Model")
mfcc_val_loss, mfcc_accuracy_beam_val, mfcc_accuracy_greedy_val, mfcc_accuracy_mrd_val = validate(mfcc_val_dataloader, mfcc_model, loss_function)
print(f"Validation Accuracy: {mfcc_accuracy_mrd_val}")



