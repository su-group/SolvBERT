import os
import pandas as pd
import torch
from rxnfp.models import SmilesClassificationModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


model_path = 'outputs/transfer_EXP'

model = SmilesClassificationModel("bert", model_path, num_labels=1,  use_cuda=torch.cuda.is_available())


test_file = pd.read_csv('data/training_files/EXP_test.csv')
test_file.columns = ['text', 'labels']

mean = test_file.labels.mean()
std = test_file.labels.std()

true_test = test_file.labels.values

preds_test = model.predict(test_file.text.values)[0]
preds_test = preds_test * std + mean


y_tests = []
r2_scores = []
rmse_scores = []


r_squared = r2_score(test_file['labels'], preds_test)
rmse = mean_squared_error(test_file['labels'], preds_test) ** 0.5
mae = mean_absolute_error(test_file['labels'], preds_test)


def make_plot(true, pred, rmse, r2_score, mae, name):
    fontsize = 12
    fig, ax = plt.subplots(figsize=(8, 8))
    r2_patch = mpatches.Patch(label="R2 = {:.3f}".format(r2_score), color="#008080")
    rmse_patch = mpatches.Patch(label="RMSE = {:.2f}".format(rmse), color="#008B8B")
    mae_patch = mpatches.Patch(label="MAE = {:.2f}".format(mae), color="#20B2AA")
    plt.xlim(-60, 10)
    plt.ylim(-60, 10)
    plt.scatter(true, pred, alpha=0.2, color="#008B8B")
    plt.plot(np.arange(-60, 10, 0.01), np.arange(-60, 10, 0.01), ls="--", c=".3")
    plt.legend(handles=[r2_patch, rmse_patch, mae_patch], fontsize=fontsize)
    ax.set_xlabel('Experimental, ΔGsolv [kcal/mol]', fontsize=fontsize)
    ax.set_ylabel('Predicted, ΔGsolv [kcal/mol]', fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)
    return fig

print(f"  R2 {r_squared:.3f},RMSE {rmse:.2f},MAE {mae:.2f}")

fig = make_plot(true_test, preds_test, rmse, r_squared, mae, ' ')
fig.savefig('data/pictures/exp_scatter.tiff', dpi=300)

def make_hist(trues, preds):
    fontsize = 12
    fig_hist, ax = plt.subplots(figsize=(8, 8))

    plt.hist(trues, bins=30, label='true',
                     facecolor='#1E90FF', histtype='bar', alpha=0.8)
    plt.hist(preds, bins=30, label='predict',
                     facecolor='#2E8B57', histtype='bar', alpha=0.6)

    plt.xlabel('ΔGsolv [kcal/mol]', fontsize=fontsize)
    plt.ylabel('amount', fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=fontsize)
    # ax.set_title(fontsize=fontsize)
    return fig_hist

fig_hist = make_hist(true_test, preds_test)
fig_hist.savefig('pictures/exp_hist.tiff', dpi=300)


# plt.show()