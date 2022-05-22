import argparse
import numpy as np

from evaluation import calculate_ppv

default_input_file='../examples/bcl/20_models/results/1798.RSR.1_32_005_025/independent0-4_monitoring0-4_number0.gz.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--predicted_file', type=str, default=default_input_file)

args = parser.parse_args()

sc_list = []
label_list = []
with open(args.predicted_file, 'r') as in_file:
	for line in in_file:
		components = line.split(',')
		pred_sc = components[0]
		label = components[1]
		sc_list.append(pred_sc)
		label_list.append(label)
		# print(f'pred_sc:{pred_sc} label:{label}')
sc_list = np.array(sc_list)
label_list = np.array(label_list)

res = calculate_ppv(label_list, sc_list)
print(res)




