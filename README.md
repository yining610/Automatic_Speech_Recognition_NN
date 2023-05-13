# Automatic_Speech_Recognition_NN

# Files and Folders:

**main.py**: Training and evaluating

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity
  -e EPOCH, --epoch EPOCH
                        number of epochs
  -f FEATURE_TYPE, --feature_type FEATURE_TYPE
                        feature type: discrete or mfcc
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size

**evaluate.py**: Load the checkpoints and evaluate

**checkpoints/**: Folder to store models' checkpoints
    ```discrete_model.pt```: Model trained by discrete feature

    ```mfcc_model.pt```: Model trained by mfcc feature

**figures/**: Folder to store loss and accuracy plot

    

# Evaluate the Trained Discrete and MFCC models

```python evaluate.py```

Output:


# Primary System

Train the model: ```python main.py -v -b 16 -f discrete -e 50```
