import torch
from rdkit import Chem

writer = Chem.SDWriter('logs/pro_mol.sdf')

mol = torch.load('logs/problematic_molecule.pt')
for cid in range(mol.GetNumConformers()):
    writer.write(mol, confId=cid)
