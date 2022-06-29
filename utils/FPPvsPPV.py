import os

in_file = '"logs/best_test_sample_scores.log"'
# in_file = '"../examples/bcl/20_models/results/1798.RSR.1_32_005_025/independent0-4_monitoring0-4_number0.gz.txt"'

out_file = '"logs/bcl_output"'
obj_function = '"FPPvsPPV(cutoff=0.1, cutoff_type=fpp_percent, parity=1)"'

# os.system(f'bcl.exe model:ComputeStatistics -input {in_file} -obj_function {obj_function} -filename_obj_function {out_file}')

os.system(f'bcl.exe model:ComputeStatistics -input {in_file} -plot_x FPR  -filename_obj_function {out_file} -image_format png')

