import torch
import random
import hashlib
import json

def get_split(num_active, num_inactive, seed, dataset_name, shrink=None):
    active_idx = list(range(num_active))
    inactive_idx = list(range(num_active, num_active + num_inactive))

    random.seed(seed)
    random.shuffle(active_idx)
    random.shuffle(inactive_idx)


    if shrink == None:
        num_active_train = round(num_active * 0.8)
        num_inactive_train = round(num_inactive * 0.8)
        num_active_valid = round(num_active * 0.1)
        num_inactive_valid = round(num_inactive * 0.1)
        num_active_test = num_active - num_active_train - num_active_valid
        num_inactive_test = round(num_inactive * 0.1)
        filename = f'data_split/{dataset_name}_seed{seed}.pt'
    else:   
        num_active_train = round(num_active * 0.8)
        num_inactive_train = 1000
        num_active_valid = round(num_active * 0.1)
        num_inactive_valid = 100
        num_active_test = num_active - num_active_train - num_active_valid
        # num_active_test = round(num_active * 0.1)
        num_inactive_test = 100
        filename = f'data_split/shrink_{dataset_name}_seed{seed}.pt'

    split_dict = {}
    split_dict['train'] = active_idx[:num_active_train]\
                          + inactive_idx[:num_inactive_train]
    split_dict['valid'] = active_idx[
                          num_active_train:num_active_train
                                           +num_active_valid] \
                          + inactive_idx[
                            num_inactive_train:num_inactive_train
                                               +num_inactive_valid]

    split_dict['test'] = active_idx[
                         num_active_train + num_active_valid
                         : num_active_train
                           + num_active_valid
                           + num_active_test] \
                         + inactive_idx[
                           num_inactive_train + num_inactive_valid
                           : num_inactive_train
                             + num_inactive_valid
                             + num_inactive_test]
    # split_dict['test'] = active_idx[
    #                      num_active_train + num_active_valid
    #                      :] \
    #                      + inactive_idx[
    #                        num_inactive_train + num_inactive_valid
    #                        :]
    num_train = len(split_dict['train'])
    num_valid = len(split_dict['valid'])
    num_test = len(split_dict['test'])
    print(f'num_train:{num_train}, num_valid:{num_valid}, num_test:{num_test}')
    
    torch.save(split_dict, filename)

    data_md5 = hashlib.md5(json.dumps(split_dict, sort_keys=True).encode('utf-8')).hexdigest()
    print(f'data_md5_checksum:{data_md5}')
    print(f'file saved at {filename}')



if __name__ == '__main__':
    dataset_info = {
        '435008':{'num_active':233, 'num_inactive':217925},
        '1798':{'num_active':187, 'num_inactive':61645},
        '435034': {'num_active':362, 'num_inactive':61394},
        '1843': {'num_active':172, 'num_inactive':301321},
        '2258': {'num_active':213, 'num_inactive':302192},
        '463087': {'num_active':703, 'num_inactive':100172},
        '488997': {'num_active':252, 'num_inactive':302054},
        '2689': {'num_active':172, 'num_inactive':319620},
        '485290': {'num_active':281, 'num_inactive':341084},
        '9999':{'num_active':37, 'num_inactive':226},
    }
    seed_list = list(range(1, 11))
    dataset_name_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
    # dataset_name_list = ['1798']
    for dataset_name in dataset_name_list:
        for seed in seed_list:
            num_actives = dataset_info[dataset_name]['num_active']
            num_inactives = dataset_info[dataset_name]['num_inactive']
            get_split(num_actives, num_inactives, seed, dataset_name, shrink=True)
