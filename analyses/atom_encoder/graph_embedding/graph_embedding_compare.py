import torch
from itertools import combinations
from torch.nn import CosineSimilarity

# 'C[C@]([H])(C1=CC=CC=C1)O', A
# 'C[C@]([H])(C1=CC=CC=C1)O', A
# '[H][C@](C1=CC=CC=C1)(O)C', B
# '[H][C@](C1=CC=CC=C1)(O)C', B
# 'C[C@]([H])(N1C=CNC1)N', C
# 'C[C@]([H])(N1C=CNC1)N', C
# 'C[C@](N)(N1C=CNC1)[H]', D
# 'C[C@](N)(N1C=CNC1)[H]', D

# graph_embedding = torch.tensor([[1.0865, -2.5481, 2.3199, 14.9122, 9.5317, -9.5077, 4.3080,
#                                  2.8809, -17.0899, -14.1018, 36.9994, 8.3337, 0.8131, -2.0581,
#                                  -2.8003, -13.9684, -18.1978, -18.4281, 17.2084, 8.9849, -14.9657,
#                                  3.1172, 32.4009, 9.4796, 5.6419, 14.7759, 5.4325, -10.5632,
#                                  8.1360, 6.9538, -20.6808, 11.0229],
#                                 [1.0865, -2.5481, 2.3199, 14.9122, 9.5317, -9.5077, 4.3080,
#                                  2.8809, -17.0899, -14.1018, 36.9994, 8.3337, 0.8131, -2.0581,
#                                  -2.8003, -13.9684, -18.1978, -18.4281, 17.2084, 8.9849, -14.9657,
#                                  3.1172, 32.4009, 9.4796, 5.6419, 14.7759, 5.4325, -10.5632,
#                                  8.1360, 6.9538, -20.6808, 11.0229],
#                                 [-12.0531, 4.0797, 6.8085, 5.4136, -2.9891, 0.1426, -2.3306,
#                                  -7.9432, -1.8055, -7.5510, 22.9870, 1.6143, -9.8032, -8.8577,
#                                  -15.3604, -28.4964, -10.2512, -30.1755, 21.4824, 0.4426, -7.4275,
#                                  21.0443, 19.0941, 14.7791, -3.2734, 4.9571, -5.7775, 5.4721,
#                                  -3.9356, -3.3048, -8.5303, -1.7977],
#                                 [-12.0531, 4.0797, 6.8085, 5.4136, -2.9891, 0.1426, -2.3306,
#                                  -7.9432, -1.8055, -7.5510, 22.9870, 1.6143, -9.8032, -8.8577,
#                                  -15.3604, -28.4964, -10.2512, -30.1755, 21.4824, 0.4426, -7.4275,
#                                  21.0443, 19.0941, 14.7791, -3.2734, 4.9571, -5.7775, 5.4721,
#                                  -3.9356, -3.3048, -8.5303, -1.7977],
#                                 [0.8058, -2.2168, 1.9143, 12.7747, 8.1207, -8.1322, 3.7419,
#                                  2.4246, -14.6294, -12.5403, 32.2326, 7.1248, 0.6934, -1.4850,
#                                  -2.4677, -12.0587, -15.6282, -16.1079, 15.1157, 7.8059, -12.8032,
#                                  2.8361, 28.6559, 8.1138, 4.9653, 12.7410, 4.7304, -9.1551,
#                                  6.9038, 6.0808, -17.9549, 9.5863],
#                                 [0.8058, -2.2168, 1.9143, 12.7747, 8.1207, -8.1322, 3.7419,
#                                  2.4246, -14.6294, -12.5403, 32.2326, 7.1248, 0.6934, -1.4850,
#                                  -2.4677, -12.0587, -15.6282, -16.1079, 15.1157, 7.8059, -12.8032,
#                                  2.8361, 28.6559, 8.1138, 4.9653, 12.7410, 4.7304, -9.1551,
#                                  6.9038, 6.0808, -17.9549, 9.5863],
#                                 [-10.3603, 3.3433, 5.7723, 4.7909, -2.4192, -0.0879, -1.8363,
#                                  -6.7584, -1.7444, -6.6019, 20.4622, 1.5154, -8.2048, -7.3523,
#                                  -13.0040, -24.3374, -8.9878, -26.0857, 18.7378, 0.6580, -6.4243,
#                                  17.9311, 17.3667, 12.5776, -2.5665, 4.5236, -4.7374, 4.3474,
#                                  -3.3332, -2.4399, -7.5688, -1.2247],
#                                 [-10.3603, 3.3433, 5.7723, 4.7909, -2.4192, -0.0879, -1.8363,
#                                  -6.7584, -1.7444, -6.6019, 20.4622, 1.5154, -8.2048, -7.3523,
#                                  -13.0040, -24.3374, -8.9878, -26.0857, 18.7378, 0.6580, -6.4243,
#                                  17.9311, 17.3667, 12.5776, -2.5665, 4.5236, -4.7374, 4.3474,
#                                  -3.3332, -2.4399, -7.5688, -1.2247]])
# smiles_list = ['A', 'B', 'C', 'D']
# # 'C[C@]([H])(C1=CC=CC=C1)O', A
# # '[H][C@](C1=CC=CC=C1)(O)C', B
# # 'C[C@]([H])(N1C=CNC1)N', C
# # 'C[C@](N)(N1C=CNC1)[H]', D

# graph_embedding = torch.tensor([[9.9795e-01, -2.4825e+00, 2.6454e+00, 1.4906e+01, 1.0401e+01,
#                                  -9.3673e+00, 4.0317e+00, 2.5838e+00, -1.6554e+01, -1.4319e+01,
#                                  3.6292e+01, 8.2951e+00, 4.8528e-01, -1.3426e+00, -3.0689e+00,
#                                  -1.3546e+01, -1.8659e+01, -1.7515e+01, 1.7084e+01, 8.4990e+00,
#                                  -1.4762e+01, 2.0697e+00, 3.2351e+01, 8.9485e+00, 5.2006e+00,
#                                  1.4096e+01, 4.5598e+00, -1.0612e+01, 8.4627e+00, 6.2798e+00,
#                                  -2.1413e+01, 1.0648e+01],
#                                 [-1.1343e+01, 4.1190e+00, 7.7695e+00, 5.3402e+00, -1.7862e+00,
#                                  2.5659e-01, -2.2165e+00, -8.2096e+00, -1.9423e+00, -8.0240e+00,
#                                  2.2333e+01, 1.3438e+00, -9.9104e+00, -7.8520e+00, -1.5534e+01,
#                                  -2.7867e+01, -9.9055e+00, -2.8755e+01, 2.1816e+01, -1.0633e-01,
#                                  -6.9887e+00, 1.9916e+01, 1.9833e+01, 1.3853e+01, -3.3299e+00,
#                                  4.3531e+00, -6.7957e+00, 4.7564e+00, -3.0305e+00, -3.7454e+00,
#                                  -9.3485e+00, -2.1555e+00],
#                                 [-8.7795e+00, 2.8111e+00, 6.2157e+00, 4.4537e+00, -1.1156e+00,
#                                  -1.0525e-02, -1.6131e+00, -6.4318e+00, -1.9609e+00, -6.4395e+00,
#                                  1.8468e+01, 1.3398e+00, -7.6011e+00, -6.0098e+00, -1.1986e+01,
#                                  -2.1721e+01, -7.9678e+00, -2.2680e+01, 1.7393e+01, 1.8991e-01,
#                                  -5.6724e+00, 1.5610e+01, 1.6557e+01, 1.0636e+01, -2.1319e+00,
#                                  3.8858e+00, -5.1175e+00, 3.2543e+00, -2.3502e+00, -2.4653e+00,
#                                  -7.7516e+00, -1.2952e+00],
#                                 [6.9527e-01, -1.9934e+00, 2.3531e+00, 1.1564e+01, 8.0855e+00,
#                                  -7.0695e+00, 3.0369e+00, 1.7332e+00, -1.2956e+01, -1.1737e+01,
#                                  2.8923e+01, 6.3490e+00, 1.2330e-01, -8.5580e-01, -2.6943e+00,
#                                  -1.0927e+01, -1.4334e+01, -1.4184e+01, 1.3891e+01, 6.4810e+00,
#                                  -1.1523e+01, 2.2538e+00, 2.6132e+01, 7.0242e+00, 4.1909e+00,
#                                  1.1104e+01, 3.4128e+00, -8.2956e+00, 6.4202e+00, 4.8389e+00,
#                                  -1.7063e+01, 8.4073e+00]])

all_graph_embedding = torch.load('graph_embedding.pt')
graph_emb_list = [all_graph_embedding[0], all_graph_embedding[200], all_graph_embedding[400], all_graph_embedding[600]]
smilesID_list = ['A', 'B', 'C', 'D']
counter = 0
with open('smiles_for_graph_embedding.txt') as f:
    all_smiles_str = f.read()
all_smiles = all_smiles_str.split('\n')
smiles_list = [all_smiles[0], all_smiles[200], all_smiles[400], all_smiles[600]]
for smiles in smiles_list:
    print(smiles)

# graph_emb_list = graph_embedding.unbind()

graph_emb_list
combs = combinations([0, 1, 2, 3], 2)

cos = CosineSimilarity(dim=-1)
for comb in combs:
    t1 = graph_emb_list[comb[0]]
    t2 = graph_emb_list[comb[1]]
    cos_sc = cos(t1, t2)
    print(f'comb:({smilesID_list[comb[0]]}, {smilesID_list[comb[1]]}) score = {cos_sc}')
