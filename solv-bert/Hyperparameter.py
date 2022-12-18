import torch
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from rxnfp.models import SmilesClassificationModel

import sklearn
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(torch.cuda.is_available())

def train():
    wandb.init()
    print("HyperParams:", wandb.config)
    model_args = {
        'wandb_project': "hyperparams_sweep_logs", 'overwrite_output_dir': True,
        'num_train_epochs': 20, 'evaluate_during_training_verbose': True,
        'gradient_accumulation_steps': 1, "warmup_ratio": 0.00,
        "train_batch_size": 16, 'regression': True, "num_labels":1,
        "fp16": False, "evaluate_during_training": True, "max_seq_length": 300,
        "config" : {

            'hidden_dropout_prob': wandb.config.dropout_rate,
            'learning_rate': wandb.config.learning_rate,
        }
            }

    model_path = 'data/out/mlm_qm_0.09'
    trained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args,
                                                use_cuda=torch.cuda.is_available())
    trained_bert.train_model(train_df, output_dir='outputs_ho',
                                eval_df=test_df, r2=sklearn.metrics.r2_score)

    wandb.log(sklearn.metrics.r2_score)



df = pd.read_csv('datas/Exp.csv')

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
eval_df, test_df = train_test_split(eval_df, test_size=0.5, random_state=42)

train_df.columns = ['text', 'labels']
test_df.columns = ['text', 'labels']


mean = train_df.labels.mean()
std = train_df.labels.std()

train_df['labels'] = (train_df['labels'] - mean) / std
test_df['labels'] = (test_df['labels'] - mean) / std

sweep_config = {
    'method': 'bayes',  # grid, random, bayes
    'metric': {
        'name': 'r2',
        'goal': 'maximize'
    },
    'parameters': {

        'learning_rate': {
            'min': 0.00001,
            'max': 0.001

        },
        'dropout_rate': {
            'min': 0.1,
            'max': 0.5
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="hyperparams_sweep")
wandb.agent(sweep_id, function=train)

