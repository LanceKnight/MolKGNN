# checking NTXcent = 0 problem

import pandas as pd
import torch
from pytorch_metric_learning.losses import NTXentLoss

df = pd.read_csv("analyses_30_loss1.366382.csv", index_col=None, header=None)
print(df)
a = df.to_numpy()
# print(a.shape)
emb = torch.tensor(a)
print(f'emb shape:{emb.shape}')

label_df = pd.read_csv("outfile_30_loss1.366382", index_col=None, header=None, sep='\t')
label_srs = label_df.iloc[:, 0]
label_lst = label_srs.tolist()
print(f'label length:{len(label_lst)}')

label = torch.tensor(label_lst)
loss = NTXentLoss()(emb, label)
print(loss)
