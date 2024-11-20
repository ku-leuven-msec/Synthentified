import json
import os

import pandas as pd

from utility.acsincome_utility import MLTYPE


def acsincome_utility_condenser(out_path):
    util_path = f'{out_path}/ACSIncome/utility'

    # creates 2 condensed tabels, one for classification and one for regression

    df_dict = {t.name: None for t in MLTYPE}
    for dirpath, dirnames, filenames in os.walk(util_path):
        for file in filenames:
            if '.csv' in file:
                continue
            file_path = os.path.join(dirpath, file)
            with open(file_path, 'r') as f:
                data = json.load(f)

            for mltype, sub_dir in data.items():
                if df_dict[mltype] is None:
                    df_dict[mltype] = pd.DataFrame(columns=['path'] + list(sub_dir.keys()))
                df_dict[mltype].loc[len(df_dict[mltype])] = [file_path] + list(sub_dir.values())

    for mltype, df in df_dict.items():
        df.to_csv(f'{util_path}/condensed_{mltype}.csv', index=False)


def patients_utility_condenser(out_path):
    util_path = f'{out_path}/patients/utility'

    # first read original to calculate relative scores (RAU as defined in our paper)
    original_path = f'{util_path}/input/train.json'
    with open(original_path, 'r') as f:
        data = json.load(f)
    original_cost = data['cost_on_original']

    df = None
    for dirpath, dirnames, filenames in os.walk(util_path):
        for file in filenames:
            if '.csv' in file:
                continue
            file_path = os.path.join(dirpath, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            data['RAU'] = data['cost_on_original'] / original_cost
            if df is None:
                df = pd.DataFrame(columns=['path'] + list(data.keys()))
            df.loc[len(df)] = [file_path] + list(data.values())
    df.to_csv(f'{util_path}/condensed.csv', index=False)


def privacy_condenser(out_path, dataset):
    priv_path = f'{out_path}/{dataset}/privacy'

    dicts = {'s_o_all': dict(), 'a_o_all': dict(), 's_h_all': dict(), 'a_h_all': dict(),
                  's_o_qid': dict(), 'a_o_qid': dict(), 's_h_qid': dict(), 'a_h_qid': dict()}

    name_mappings = {
        's': {'o_all_True': 's_o_all', 'h_all_True': 's_h_all', 'o_all_False': 's_o_qid', 'h_all_False': 's_h_qid',
              'o_closest_True': 's_o_qid', 'h_closest_True': 's_h_qid', 'o_closest_False': 's_o_qid',
              'h_closest_False': 's_h_qid'},
        'a': {'o_all_True': 'a_o_all', 'h_all_True': 'a_h_all', 'o_all_False': 'a_o_qid', 'h_all_False': 'a_h_qid',
              'o_closest_True': 'a_o_qid', 'h_closest_True': 'a_h_qid', 'o_closest_False': 'a_o_qid',
              'h_closest_False': 'a_h_qid'}}

    counter = 0

    for dirpath, dirnames, filenames in os.walk(priv_path):
        for file in filenames:
            if '.csv' in file:
                continue
            counter += 1
            if not counter % 1000:
                print(counter)
            #if counter == 2000:
            #    return
            file_path = os.path.join(dirpath, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'FAILED_RG' in data:
                del data['FAILED_RG']

            # is this dataset synthetic or anonymous?
            dataset_type = 'a' if 'Anonymized' in file_path else 's'
            is_original = 'input' in file_path

            tmp_dicts = {'s_o_all': dict(), 'a_o_all': dict(), 's_h_all': dict(), 'a_h_all': dict(),
                         's_o_qid': dict(), 'a_o_qid': dict(), 's_h_qid': dict(), 'a_h_qid': dict()}

            # max nesting of a metric is a dict of dicts
            for metric_name, metric_data in data.items():
                for submetric_name, submetric_data in metric_data.items():

                    splitted_submetric_name = submetric_name.split('_')
                    new_name = '_'.join(splitted_submetric_name[:-3])
                    df_indicator = '_'.join(splitted_submetric_name[-3:])
                    if not is_original:
                        this_dict = tmp_dicts[name_mappings[dataset_type][df_indicator]]
                    else:
                        this_dict = tmp_dicts[name_mappings['a'][df_indicator]]
                        this_dict2 = tmp_dicts[name_mappings['s'][df_indicator]]

                    if isinstance(submetric_data, dict):

                        # exception for 'nearest_neighbors_aia'
                        new_metric_name = metric_name
                        if new_metric_name == 'nearest_neighbors_aia':
                            new_metric_name = f'{new_metric_name}_{"_".join(splitted_submetric_name[-2:])}'

                        # another dict is nested
                        for subsubmetric_name, subsubmetric_data in submetric_data.items():
                            tmp_new_name = f'{new_metric_name}__{new_name}__{subsubmetric_name}'
                            this_dict[tmp_new_name] = subsubmetric_data
                            if is_original:
                                this_dict2[tmp_new_name] = subsubmetric_data
                    else:
                        # we have a value
                        this_dict[new_name] = submetric_data
                        if is_original:
                            this_dict2[new_name] = submetric_data
            for name, tmp_dict in tmp_dicts.items():
                if len(tmp_dict):
                    dicts[name][file_path] = tmp_dict

    for name, this_dict in dicts.items():
        df = pd.DataFrame.from_dict(this_dict, orient='index')
        df.index.name = 'path'
        df.to_csv(f'{priv_path}/condensed_{name}.csv')


def statistical_utility_condenser(out_path, dataset):
    util_path = f'{out_path}/{dataset}/queries'

    df = None
    for dirpath, dirnames, filenames in os.walk(util_path):
        for file in filenames:
            if '.csv' in file:
                continue
            file_path = os.path.join(dirpath, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            if df is None:
                df = pd.DataFrame(columns=['path'] + list(data.keys()))
            df.loc[len(df)] = [file_path] + list(data.values())

    df.to_csv(f'{util_path}/condensed.csv', index=False)


def arx_statistics_condenser(out_path, dataset):
    util_path = f'{out_path}/{dataset}/arx_stats'

    df = None
    for dirpath, dirnames, filenames in os.walk(util_path):
        for file in filenames:
            if 'condensed' in file:
                continue
            file_path = os.path.join(dirpath, file)
            data = pd.read_csv(file_path)
            if df is None:
                df = pd.DataFrame(columns=['path'] + list(data.columns))
            df.loc[len(df)] = [file_path] + data.iloc[0].tolist()
    df.to_csv(f'{util_path}/condensed.csv', index=False)

if __name__ == '__main__':
    output_path = '../output'
    datasets = ['ACSIncome', 'patients']
    utility_condensers = {'ACSIncome': acsincome_utility_condenser, 'patients': patients_utility_condenser}
    privacy_condensers = {'ACSIncome': privacy_condenser, 'patients': privacy_condenser}
    statistical_utility_condensers = {'ACSIncome': statistical_utility_condenser, 'patients': statistical_utility_condenser}
    arx_statistics_condensers = {'ACSIncome': arx_statistics_condenser, 'patients': arx_statistics_condenser}

    for dataset in datasets:
        print(dataset)
        utility_condensers[dataset](output_path)
        privacy_condensers[dataset](output_path, dataset)
        statistical_utility_condensers[dataset](output_path, dataset)
        arx_statistics_condensers[dataset](output_path, dataset)
