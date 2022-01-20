
# argv 1 is the input smiles csv file
# argv 2 is the training csv file that should not be contained in input csv file

import sys
import pandas as pd
from tqdm import tqdm


input = '../../dataset/qsar/raw/1798_inactives.smi'  # sys.argv[1]
training_inactive_smiles = '../../dataset/qsar/raw/435034_inactives.smi'  # sys.argv[2]
training_active_smiles = '../../dataset/qsar/raw/435034_actives.smi'
input_df = pd.read_csv(input, header=None, sep='\t')[0]
train_df1 = pd.read_csv(training_active_smiles, header=None, sep='\t')[0]
train_df2 = pd.read_csv(training_inactive_smiles, header=None, sep='\t')[0]

train_df = train_df1.append(train_df2)
# print(type(train_df.tolist()))

output_lst = []
dup_counter = 0
progress = tqdm(input_df, desc='process')
for smiles in progress:
    progress.set_description(f'dup counter: {dup_counter}')
    if smiles in train_df:
        dup_counter += 1
    # print(smiles)
    else:
        output_lst.append(smiles)


output_df = pd.DataFrame(output_lst)
# print(output_df)
output_df.to_csv('../../dataset/connect_aug/raw/smiles.csv', index=False, header=False)
