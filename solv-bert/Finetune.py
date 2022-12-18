from sklearn.model_selection import train_test_split
import pkg_resources
import torch
from rxnfp.models import SmilesClassificationModel
import pandas as pd

import os

import wandb
wandb.login(key='cdfffe511a8a7c87b9781d7c2a63578243b22d80')
wandb.init(project="ho_exp_finetune", entity="aichem")

wandb.config = {
  "learning_rate": 0.000001,
  "epochs": 100,
  "batch_size": 128
}
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(torch.cuda.is_available())


train_df = pd.read_csv('data/training_files/EXP_train.csv')
eval_df = pd.read_csv('data/training_files/EXP_train.csv')


train_df.columns = ['text', 'labels']
eval_df.columns = ['text', 'labels']


mean = train_df.labels.mean()
std = train_df.labels.std()
train_df['labels'] = (train_df['labels'] - mean) / std
eval_df['labels'] = (eval_df['labels'] - mean) / std

model_args = {
     'num_train_epochs': 20, 'overwrite_output_dir': True,
    'learning_rate': 0.00008, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": True, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': 0.4 },
    'wandb_project':"ho_exp_finetune"
}


model_path = 'data/out/mlm_qm_0.09'



dg_bert = SmilesClassificationModel("bert", model_path, num_labels=1,
                                       args=model_args, use_cuda=torch.cuda.is_available())



dg_bert.train_model(train_df, output_dir="data/output/exp", eval_df=eval_df)

"""save the best_model manually"""




