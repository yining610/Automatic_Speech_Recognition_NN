# Automatic_Speech_Recognition_NN

# Files and Folders:

**main.py**: Training and evaluating

optional arguments:
```
  -v, --verbose         increase output verbosity
  -e EPOCH, --epoch EPOCH
                        number of epochs
  -f FEATURE_TYPE, --feature_type FEATURE_TYPE
                        feature type: discrete or mfcc
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
 ```                       

**evaluate.py**: Load the checkpoints and evaluate the models

**discrete_test_result.json**: 393 testing result by Discrete model

**mfcc_test_reesult.json**: 393 testing result by MFCC model

**/checkpoints**: Folder to store models' checkpoints

```./checkpoints/discrete_model.pt```: Model trained by discrete feature

```./checkpoints/mfcc_model.pt```: Model trained by mfcc feature


**/figures**: Folder to store loss and accuracy plots

```./figures/discrete_accuracy.png```: Training and validating accuracy using Greedy and Beam search for Discrete model

```./figures/mrd_accuracy.png```: Validating accuracy using Minimum CTCLoss Decoding strategy for Discrete model

```./figures/discrete_loss.png```: Training and validating CTC loss for Discrete model

```./figures/mfcc_accuracy.png```: Training and validating accuracy using Greedy and Beam search for MFCC model

```./figures/mfcc_accuracy.png```: Validating accuracy using Minimum CTCLoss Decoding strategy for MFCC model

```./figures/mfcc_loss.png```: Training and validating CTC loss for MFCC model

# Evaluate the Trained Discrete and MFCC models

```python evaluate.py``` loads saved model and evaluate accuracy using the same validation datasets used in training.

Output:

```
Evaluating on Validation Set Using Minimum CTC Loss Decoder For Discrete Model
Validation Greedy Search Decoded Words: ['iny', 'ttm', 'bouh', 'oway', 'every', 'soes', 'st', 'nny']
Validationo Beam Search Decoded Words: ['iny', 'ttm', 'bouh', 'oway', 'every', 'soes', 'st', 'nny']
Validation CTC Decoded Words: ['many', 'often', 'about', 'away', 'every', 'sometimes', 'system', 'many']
original words: ['many', 'often', 'about', 'away', 'every', 'sometimes', 'extra', 'money']
Validation Accuracy: 0.775

Evaluating on Validation Set Using Minimum CTC Loss Decoder For MFCC Model
Validation Greedy Search Decoded Words: ['', '', '', '', '', '', 't', '']
Validationo Beam Search Decoded Words: ['', '', '', '', '', '', 't', '']
Validation CTC Decoded Words: ['even', 'often', 'about', 'eating', 'every', 'enough', 'after', 'over']
original words: ['many', 'often', 'about', 'away', 'every', 'sometimes', 'extra', 'money']
Validation Accuracy: 0.525
```


# Primary System

Train the model: ```python main.py -v -b 16 -f discrete -e 50```

Outputs:  ```./figuress/discrete_accuracy.png ./figures/discrete_loss.png ./figures/discrete_mrd_accuracy.png```, ```discrete_test_result.json```, ```./checkpoints/discrete_model.pt```

Snapshot:


Dataset setting:
1. Training size: 758
2. Validating size: 40
3. Testing size: 393

Hyperparameter Settings: 
1. seed for ```random_split```: 0
2. model: LSTM
3. Embedding size: 40
4. hidden state size: 256
5. number layers: 2
6. learning rate: 5e-3
7. silence token id: 23
8. blank token id: 24
9. pad token id: 25


# Contrastive System

Train the model: ```python main.py -v -b 16 -f mfcc -e 50```

Dataset setting:
1. Training size: 758
2. Validating size: 40
3. Testing size: 393

Hyperparameter Settings:
1. seed for ```random_split```: 0
2. model: LSTM
3. input size: 40
4. hidden state size: 256
5. number layers: 2
6. learning rate: 5e-3
7. silence token id: 23
8. blank token id: 24
9. pad token id: 25

