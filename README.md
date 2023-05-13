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


**/figures**: Folder to store loss and accuracy plot

```./figures/discrete_accuracy.png```: Training and validating accuracy using Greedy and Beam search for Discrete Model

```./figures/mrd_accuracy.png```: Validating accuracy using Minimum CTCLoss Decoding strategy for Discrete Model

```./figures/discrete_loss.png```: Training and validating CTC loss for Discrete Model

```./figures/mfcc_accuracy.png```: Training and validating accuracy using Greedy and Beam search for MFCC model

```./figures/mfcc_accuracy.png```: Validating accuracy using Minimum CTCLoss Decoding strategy for MFCC Model

```./figures/mfcc_loss.png```: Training and validating CTC loss for MFCC Model

# Evaluate the Trained Discrete and MFCC models

```python evaluate.py``` loads saved model and evaluate accuracy using the same validation datasets used in training.

Output:


# Primary System

Train the model: ```python main.py -v -b 16 -f discrete -e 50```

Hyperparameter Settings: 
1. seed for ```random_split```: 0
2. model: LSTM
3. input size: 40
4. hidden state size: 256
5. number layers: 2
6. learning rate: 5e-3

