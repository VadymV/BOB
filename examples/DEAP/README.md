Examples on how different models can be trained.

### Data setup:
- Download data: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- Extract ```data_preprocessed_python.zip```
- Create a folder (I will refer to this folder as PROJECT_FOLDER) and place the extracted data into this folder.

### Training:
- The script ```split_data.py``` preprocesses the data and splits the data into train/validation/test sets.
- The script ```train_ssl.py``` trains a representation model in a self-supervised way without labels.
- There are additional examples on how to train other models.
- The training process can be seen on the W&B (https://wandb.ai/site) dashboard (you need to create an account).
