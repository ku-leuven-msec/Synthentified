import os

import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from threadpoolctl import ThreadpoolController

controller = ThreadpoolController()
controller.limit(limits=1)

import glob
import json
import subprocess
from multiprocessing import Pool

import pandas as pd
from ctgan import TVAE, CTGAN
from tqdm import tqdm

from utility.statistical_utility import partition_num_data, get_encoders, get_uniform_probabilities, get_encoded, \
    generate_random_queries, evaluate_queries_original_domain, estimate_queries
from DistanceMetrics import CatMetric, NumMetric, DistanceBuilder, sample_to_original_domain
from utility.acsincome_utility import get_ml_utility, MLTYPE
from gen_syn import generate_synthetic_sdv, generate_synthetic_synthcity
from privacy.metrics import StructuredPrivacyMetrics
from utility.patients_utility import get_patients_utility
from privacy.helper import make_meta_data_for_privacy_eval, get_distance_and_indices, concat_generalized_columns


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


def train_test_splits(params):
    # read original data, sample and write back
    datasets = params['dataset']
    datasets_to_evaluate = params['datasets_to_evaluate']
    for dataset_name in tqdm(datasets_to_evaluate, total=len(datasets_to_evaluate), desc='Train/test splits'):
        dataset_params = datasets[dataset_name]
        inputData = pd.read_csv(dataset_params['path'], dtype=str)
        inputData = inputData[dataset_params['columns']]
        train = inputData.sample(frac=1 - params['test_ratio'])
        test = inputData.drop(train.index)
        outpath = params['output_path'] + dataset_name + '/datasets/input/'
        os.makedirs(outpath, exist_ok=True)
        train.to_csv(outpath + 'train.csv', index=False)
        test.to_csv(outpath + 'test.csv', index=False)


def create_anonymized_datasets(params):
    datasets_to_evaluate = params['datasets_to_evaluate']
    print_all = str(params['print_all_anonymous'])
    for dataset_name in tqdm(datasets_to_evaluate, total=len(datasets_to_evaluate), desc='Anonymize'):
        try:
            print(f'Anonymizing {dataset_name}')
            train_path = params['output_path'] + dataset_name + '/datasets/input/train.csv'
            outpath = params['output_path'] + dataset_name + '/datasets/Anonymized/'
            arx_stats_outpath = params['output_path'] + dataset_name + '/arx_stats/Anonymized/'
            os.makedirs(outpath, exist_ok=True)
            # run java code with following param order inputPath, testName, outPath
            result = subprocess.run(
                ['java', '-jar', '../intellij/arx.jar', train_path, dataset_name, outpath, print_all,
                 arx_stats_outpath])
        except Exception as e:
            raise Exception(f"Anonymization failed for {dataset_name}") from e


def create_synthetic_datasets(params):
    datasets = params['dataset']
    datasets_to_evaluate = params['datasets_to_evaluate']
    for dataset_name in tqdm(datasets_to_evaluate, total=len(datasets_to_evaluate), desc='Generate synthetic'):
        try:
            print(f'Creating synthetic {dataset_name}')
            dataset_params = datasets[dataset_name]
            o_data = pd.read_csv(params['output_path'] + dataset_name + '/datasets/input/train.csv', dtype=str)
            num = dataset_params['num']
            outpath = params['output_path'] + dataset_name + '/datasets/Synthetic/'
            os.makedirs(outpath, exist_ok=True)
            synth_params = dataset_params['synth_params']
            for model in params['synthetic_models']:
                if params['synthetic_model_library'][model] == 'sdv':
                    model_obj = params['synth_model_obj'][model]
                    model_params = synth_params[model]
                    generate_synthetic_sdv(o_data[synth_params['synth_columns']], num, model_obj, model_params, outpath,
                                           params['synth_samples'], False, synth_params['synth_post_processor'])
                if params['synthetic_model_library'][model] == 'synthcity':
                    model_name = params['synth_model_obj'][model]
                    qid = dataset_params['qid']
                    model_params = synth_params[model]
                    target_col = synth_params['target']
                    generate_synthetic_synthcity(o_data[synth_params['synth_columns']], num, model_name, qid,
                                                 target_col, model_params, outpath, params['synth_samples'],
                                                 synth_params['synth_post_processor'])
        except Exception as e:
            raise Exception(f"Synthetic data failed for {dataset_name}") from e


def measure_utility(params):
    # we will try to parallelize this for each dataset type (patients or ACSIncome)
    # first create a list of all filepaths that need to be calculated,
    # this must include all anonymized, synthetic and original datasets of a certain type

    datasets_to_evaluate = params['datasets_to_evaluate']
    datasets = params['dataset']
    types_to_evaluate = params['types_to_evaluate']
    for dataset_name in tqdm(datasets_to_evaluate, total=len(datasets_to_evaluate), desc='Measuring utility:'):
        dataset_params = datasets[dataset_name]
        paths = []
        if 'original' in types_to_evaluate:
            paths.append(params['output_path'] + dataset_name + '/datasets/input/train.csv')
        if 'synthetic' in types_to_evaluate:
            paths += glob.glob(params['output_path'] + dataset_name + '/datasets/Synthetic/*_syn.csv')
        if 'anonymous' in types_to_evaluate:
            # filter to not include settings and configs files
            for p in glob.glob(params['output_path'] + dataset_name + '/datasets/Anonymized/**/*.csv', recursive=True):
                f_name = p.split('/')[-1]
                if 'settings.csv' != f_name and 'configs.csv' != f_name:
                    paths.append(p)
        # run parallel
        if not debug_mode:
            with Pool(processes=dataset_params['util_params']['cores'], initializer=util_parallel_initializer,
                      initargs=(params,)) as p:
                list(tqdm(p.imap_unordered(dataset_params['util_calculator'], paths), total=len(paths),
                          desc=f'{dataset_name} utility'))
        else:
            util_parallel_initializer(params)
            for j in tqdm(paths, total=len(paths), desc=f'{dataset_name} utility'):
                dataset_params['util_calculator'](j)


# needed as it is not guarantied that both possible targets in the synthetic data say that it is above 50k
def ACSIncome_post_processor(df):
    df['PINCP class'] = (df['PINCP'] > 50000).astype(int)
    return df


# makes sure that each core can read the params dict when needed
shared_params = None


def util_parallel_initializer(params):
    global shared_params
    shared_params = params


def ACSIncome_util(path):
    try:
        acs_util_params = shared_params['dataset']['ACSIncome']['util_params']
        out_path = f'{shared_params["output_path"]}ACSIncome/'
        # all needed data can be found relative to the given path
        train_path = path
        train = pd.read_csv(train_path, dtype=str)
        test_path = out_path + 'datasets/input/test.csv'
        test = pd.read_csv(test_path, dtype=str)
        # keep same folder structures as the datasets
        out_file = path.replace('datasets', 'utility').replace('.csv', '.json')
        os.makedirs('/'.join(out_file.split('/')[:-1]), exist_ok=True)

        scores = {}
        for ml_type in acs_util_params['ml_types']:
            score = get_ml_utility(train, test, ml_type, path, acs_util_params['generalize'],
                                   acs_util_params['drop_suppressed'])
            scores[ml_type.name] = score
        with open(out_file, 'w') as f:
            json.dump(scores, f)
    except Exception as e:
        raise Exception(f"ACSIncome utility failed for {path}") from e


def patients_util(path):
    try:
        out_path = f'{shared_params["output_path"]}patients/'
        orig_path = out_path + 'datasets/input/train.csv'
        sampling = shared_params['dataset']['patients']['util_params']['sampling']
        # keep same folder structures as the datasets
        out_file = path.replace('datasets', 'utility').replace('.csv', '.json')

        os.makedirs('/'.join(out_file.split('/')[:-1]), exist_ok=True)

        df_patients_test = pd.read_csv(path, dtype=str)
        df_facilities = pd.read_csv('../Datasets/Facilities-Extended.csv')
        df_patients_orig = pd.read_csv(orig_path, dtype=str)
        df_zipcodes = pd.read_csv('../Datasets/ZIPCODES.csv',
                                  dtype={'zip': 'str', 'latitude': 'float', 'longitude': 'float'})

        output = get_patients_utility(df_patients_test, df_facilities, df_patients_orig, df_zipcodes, sampling=sampling)

        with open(out_file, 'w') as f:
            json.dump(output, f)
    except Exception as e:
        raise Exception(f"Patients utility failed for {path}") from e


def measure_additional_metrics(params):
    # adds classification metric to ACSIncome dataset arx statistics and gen level for each qid
    datasets_to_evaluate = params['datasets_to_evaluate']
    datasets = params['dataset']
    types_to_evaluate = params['types_to_evaluate']
    for dataset_name in tqdm(datasets_to_evaluate, total=len(datasets_to_evaluate),
                             desc='Measuring statistical utility:'):
        dataset_params = datasets[dataset_name]
        paths = []
        if 'anonymous' in types_to_evaluate:
            # filter to not include settings and configs files
            for p in glob.glob(params['output_path'] + dataset_name + '/datasets/Anonymized/**/*.csv', recursive=True):
                f_name = p.split('/')[-1]
                if 'settings.csv' != f_name and 'configs.csv' != f_name:
                    paths.append(p)

        init_data = {'qid': dataset_params['qid']}

        # run parallel
        if not debug_mode:
            with Pool(processes=dataset_params['util_params']['cores'],
                      initializer=statistical_util_parallel_initializer,
                      initargs=(params, init_data)) as p:
                list(tqdm(p.imap_unordered(calculate_additional_metrics, paths), total=len(paths),
                          desc=f'{dataset_name} additional metrics'))
        else:
            statistical_util_parallel_initializer(params, init_data)
            for j in tqdm(paths, total=len(paths), desc=f'{dataset_name} additional metrics'):
                calculate_additional_metrics(j)


def calculate_additional_metrics(path):
    out_file = path.replace('datasets', 'arx_stats')
    statistics = pd.read_csv(out_file)
    qids = shared_init_data['qid']

    if 'ACSIncome' in path:
        # calculate for each ACSIncome dataset the classification metric for binary and multiclass,
        # see: https://dl.acm.org/doi/10.1145/775047.775089
        # modification: we see suppressed also as its own group
        p_data = pd.read_csv(path, dtype=str)

        p_data['target1'] = p_data['PINCP'].astype(int) > 50000
        p_data['target2'] = pd.cut(p_data['PINCP'].astype(int), bins=[0, 20000, 100000, np.inf],
                                       right=False, include_lowest=True).astype(str)

        grouped = p_data.groupby(by=qids)
        cm1 = 0
        cm2 = 0
        for key, group in grouped:
            majority1 = group['target1'].mode()[0]
            majority2 = group['target2'].mode()[0]
            cm1 += sum(group['target1'] != majority1)
            cm2 += sum(group['target2'] != majority2)

        cm1 /= len(p_data)
        cm2 /= len(p_data)
        statistics['cm_binary'] = cm1
        statistics['cm_multiclass'] = cm2

    # add gen depth for each qid
    # test if the given dataset is from within the anonymized folder by checking if a specific file exists
    is_anonymized = os.path.isfile(f'{os.path.dirname(path)}/settings.csv')
    if is_anonymized:
        # retrieve generalization levels for each QID from the file name
        levels = os.path.basename(path)[:-4].split('_')
    else:
        levels = [-1] * len(qids)
    for qid, level in zip(qids, levels):
        statistics[f'{qid} gen level'] = level

    statistics.to_csv(out_file, index=False)


def measure_statistical_utility(params):
    # we will try to parallelize this for each dataset type (patients or ACSIncome)
    # first create a list of all filepaths that need to be calculated,
    # this must include all anonymized, synthetic and original datasets of a certain type

    datasets_to_evaluate = params['datasets_to_evaluate']
    datasets = params['dataset']
    types_to_evaluate = params['types_to_evaluate']
    for dataset_name in tqdm(datasets_to_evaluate, total=len(datasets_to_evaluate),
                             desc='Measuring statistical utility:'):
        dataset_params = datasets[dataset_name]
        paths = []
        if 'synthetic' in types_to_evaluate:
            paths += glob.glob(params['output_path'] + dataset_name + '/datasets/Synthetic/*_syn.csv')
        if 'anonymous' in types_to_evaluate:
            # filter to not include settings and configs files
            for p in glob.glob(params['output_path'] + dataset_name + '/datasets/Anonymized/**/*.csv', recursive=True):
                f_name = p.split('/')[-1]
                if 'settings.csv' != f_name and 'configs.csv' != f_name:
                    paths.append(p)

        init_data = dict()
        init_data['original_path'] = params['output_path'] + dataset_name + '/datasets/input/train.csv'

        original_df = pd.read_csv(init_data['original_path'], dtype=str)

        # Scott's normal reference rule with fixed n value of 1000
        bucket_width = {col: 3.49 + original_df[col].astype(float).std() * pow(1000, -1 / 3) for col in
                        dataset_params['floats']}
        bucket_amounts = {
            col: round((original_df[col].astype(float).max() - original_df[col].astype(float).min()) / width) for
            col, width in
            bucket_width.items()}

        hierarchies = {qi: pd.read_csv(f'../Hierarchies/{dataset_name}/{qi}.csv', sep=';', decimal=',',
                                       header=None, dtype=str) for qi in dataset_params['qid']}

        columns_dict = {
            'all_columns': dataset_params['columns'],
            'qid_columns': dataset_params['qid'],
            'non_qid_columns': list(set(dataset_params['columns']) - set(dataset_params['qid']))
        }

        # bucketize big numeric domains
        partitioned_original_df, buckets = partition_num_data(original_df, bucket_amounts)
        # generated encoders + the encoding of the original data for all categorical data and bucketized data
        encoders, cat_options = get_encoders(partitioned_original_df, hierarchies)

        # generate a set of queries for 3 situations: all_columns, qid_columns, non_qid_columns
        queries_dict = {key: generate_random_queries(params['query_amount'], columns, cat_options) for key, columns in
                        columns_dict.items()}

        # get the original data after applying the encoders
        orig_df_encoded = get_encoded(partitioned_original_df, encoders)

        # evaluate what the real query results are on original data queries for each situation
        real_result_dict = {
            key: np.array(evaluate_queries_original_domain(orig_df_encoded[columns_dict[key]], queries_dict[key])) for
            key in columns_dict.keys()}

        # pre-calculate distributions for generalized data, assume uniform distributions
        contains_dict, probability_dict = get_uniform_probabilities(hierarchies, encoders)

        init_data['real_result_dict'] = real_result_dict
        init_data['contains_dict'] = contains_dict
        init_data['probability_dict'] = probability_dict
        init_data['encoders'] = encoders
        init_data['columns_dict'] = columns_dict
        init_data['queries_dict'] = queries_dict
        init_data['qid'] = dataset_params['qid']
        init_data['buckets'] = buckets

        # run parallel
        if not debug_mode:
            with Pool(processes=dataset_params['util_params']['cores'],
                      initializer=statistical_util_parallel_initializer,
                      initargs=(params, init_data)) as p:
                list(tqdm(p.imap_unordered(calculate_statistical_utility, paths), total=len(paths),
                          desc=f'{dataset_name} utility'))
        else:
            statistical_util_parallel_initializer(params, init_data)
            for j in tqdm(paths, total=len(paths), desc=f'{dataset_name} utility'):
                calculate_statistical_utility(j)


def calculate_statistical_utility(path):
    # keep same folder structures as the datasets
    out_file = path.replace('datasets', 'queries').replace('.csv', '.json')
    os.makedirs('/'.join(out_file.split('/')[:-1]), exist_ok=True)

    p_data = pd.read_csv(path, dtype=str)

    p_data = get_encoded(partition_num_data(p_data, shared_init_data['buckets'])[0], shared_init_data['encoders'])

    query_results_dict = dict()
    qid = shared_init_data['qid']
    prob_dict = shared_init_data['probability_dict']
    contains_dict = shared_init_data['contains_dict']
    for situation in shared_init_data['columns_dict'].keys():
        columns = shared_init_data['columns_dict'][situation]
        data = p_data[columns]
        queries = shared_init_data['queries_dict'][situation]

        result = np.array(estimate_queries(data, contains_dict, prob_dict, queries, qid, columns))
        real_result = shared_init_data['real_result_dict'][situation]

        error = np.divide(np.abs(real_result - result), real_result,
                          out=np.full(real_result.shape, np.NAN, dtype=float), where=real_result != 0)
        avg_error = np.nanmean(error)

        query_results_dict[situation] = avg_error

    with open(out_file, 'w') as f:
        json.dump(query_results_dict, f, cls=NpEncoder)


shared_init_data = None


def statistical_util_parallel_initializer(params, init_data):
    global shared_params, shared_init_data
    shared_params = params
    shared_init_data = init_data


def measure_privacy(params):
    datasets_to_evaluate = params['datasets_to_evaluate']
    datasets = params['dataset']
    types_to_evaluate = params['types_to_evaluate']

    for dataset_name in tqdm(datasets_to_evaluate, total=len(datasets_to_evaluate), desc='Measuring privacy'):
        original = pd.read_csv(params['output_path'] + dataset_name + '/datasets/input/train.csv', dtype=str)
        holdout = pd.read_csv(params['output_path'] + dataset_name + '/datasets/input/test.csv', dtype=str)
        original_full = pd.read_csv(datasets[dataset_name]['path'], dtype=str)[datasets[dataset_name]['columns']]

        # can be reused in each process
        meta_data_dict, hierarchies_dict, db_dict = make_meta_data_for_privacy_eval(original_full, params,
                                                                                    dataset_name, True)

        # distance builder used for o to o, can be passed to each process as well
        builder = DistanceBuilder(**db_dict)

        # distance results on original and holdout can be re-used in each process
        qid = datasets[dataset_name]['qid']
        data_dict = {'o': original, 'h': holdout}
        data_dist = get_distance_and_indices(data_dict, 'o', 'o', builder, qid=qid, n_neighbors=params['n_neighbors'])
        data_dist2 = get_distance_and_indices(data_dict, 'h', 'h', builder, qid=qid, n_neighbors=params['n_neighbors'])
        # join both data_dist lists of dicts
        for i, d in enumerate(data_dist):
            d.update(data_dist2[i])

        del original, holdout, original_full, data_dist2, data_dict

        pre_calc_data = {
            'meta_data': meta_data_dict,
            'hierarchies': hierarchies_dict,
            'builder': builder,
            'data_dist': data_dist,
            'qid': datasets[dataset_name]['qid']
        }

        # anonymous and synthetic dataset to run
        paths = []
        if 'original' in types_to_evaluate:
            paths.append(params['output_path'] + dataset_name + '/datasets/input/train.csv')
        if 'synthetic' in types_to_evaluate:
            paths += glob.glob(params['output_path'] + dataset_name + '/datasets/Synthetic/*_syn.csv')
        if 'anonymous' in types_to_evaluate:
            # filter to not include settings and configs files
            for p in glob.glob(params['output_path'] + dataset_name + '/datasets/Anonymized/**/*.csv', recursive=True):
                f_name = p.split('/')[-1]
                if 'settings.csv' != f_name and 'configs.csv' != f_name:
                    paths.append(p)

        # run parallel
        if not debug_mode:
            with open(f"../output/{dataset_name}/failed.txt", "w") as failed_f:
                failed_count = 0
                with Pool(processes=datasets[dataset_name]['privacy_params']['cores'],
                          initializer=privacy_parallel_initializer,
                          initargs=(params, pre_calc_data)) as p:
                    for result in tqdm(p.imap_unordered(calculate_priv, paths), total=len(paths),
                                       desc=f'{dataset_name} privacy'):
                        if result:
                            failed_count += 1
                            failed_f.write(f"{result.split(' ')[-1]}\n")
                            print(f"Currently failed: {failed_count}, {result}")
        else:
            privacy_parallel_initializer(params, pre_calc_data)
            for j in tqdm(paths, total=len(paths), desc=f'{dataset_name} privacy'):
                calculate_priv(j)


def measure_crashed_privacy(params):
    """Rerun {'True-o-gmm_log_likelihood', 'True-h-gmm_log_likelihood'} for failed datasets until no failure.
    these are the only metrics that sometimes crash
    This is done be redoing the sampling over and over again in the hope of not failing."""
    datasets_to_evaluate = params['datasets_to_evaluate']
    datasets = params['dataset']

    for dataset_name in tqdm(datasets_to_evaluate, total=len(datasets_to_evaluate), desc='Measuring failed privacy'):
        original_full = pd.read_csv(datasets[dataset_name]['path'], dtype=str)[datasets[dataset_name]['columns']]

        # can be reused in each process
        meta_data_dict, hierarchies_dict, db_dict = make_meta_data_for_privacy_eval(original_full, params,
                                                                                    dataset_name, True)

        pre_calc_data = {
            'meta_data': meta_data_dict,
            'hierarchies': hierarchies_dict,
            'qid': datasets[dataset_name]['qid']
        }

        # anonymous and synthetic dataset to run
        with open(f'{params["output_path"]}{dataset_name}/failed.txt', 'r') as f:
            paths = f.read().splitlines()

        # run parallel
        if not debug_mode:
            with Pool(processes=datasets[dataset_name]['privacy_params']['cores'],
                      initializer=privacy_parallel_initializer,
                      initargs=(params, pre_calc_data)) as p:
                list(tqdm(p.imap_unordered(calculate_crashed_priv, paths), total=len(paths),
                          desc=f'{dataset_name} crashed privacy'))
        else:
            privacy_parallel_initializer(params, pre_calc_data)
            for j in tqdm(paths, total=len(paths), desc=f'{dataset_name} crashed privacy'):
                calculate_crashed_priv(j)


shared_pre_calc_data = None


def privacy_parallel_initializer(params, pre_calc_data):
    global shared_params, shared_pre_calc_data
    shared_params = params
    shared_pre_calc_data = pre_calc_data


def calculate_priv(path):
    try:
        # keep same folder structures as the datasets
        out_file = path.replace('datasets', 'privacy').replace('.csv', '.json')
        os.makedirs('/'.join(out_file.split('/')[:-1]), exist_ok=True)

        original_path = f'{path.split("datasets")[0]}datasets/input/train.csv'
        holdout_path = f'{path.split("datasets")[0]}datasets/input/test.csv'

        o_data = pd.read_csv(original_path, dtype=str)
        h_data = pd.read_csv(holdout_path, dtype=str)
        p_data = pd.read_csv(path, dtype=str)

        data_dict = {
            'o': o_data,
            'p': p_data,
            'h': h_data
        }

        # first calculate all distances needed, o_o and h_h is already calculated in shared_pre_calc_data[data_dist]
        dataset1_names = ['o', 'h', 'p']
        distance_dict, indices_dict, distance_dict_qid, indices_dict_qid = dict(), dict(), dict(), dict()
        if distance_based_only or not ml_only:
            for n1 in dataset1_names:
                d_dict, i_dict, \
                    d_dict_qid, i_dict_qid = get_distance_and_indices(data_dict, n1, 'p',
                                                                      shared_pre_calc_data['builder'],
                                                                      qid=shared_pre_calc_data['qid'],
                                                                      n_neighbors=shared_params['n_neighbors'])
                distance_dict.update(d_dict)
                indices_dict.update(i_dict)
                distance_dict_qid.update(d_dict_qid)
                indices_dict_qid.update(i_dict_qid)

        # add o_o and h_h from shared_pre_calc_data[data_dist]
        distance_dict.update(shared_pre_calc_data['data_dist'][0])
        indices_dict.update(shared_pre_calc_data['data_dist'][1])
        distance_dict_qid.update(shared_pre_calc_data['data_dist'][2])
        indices_dict_qid.update(shared_pre_calc_data['data_dist'][3])

        # all data must be in original domain so sampling trick must apply on anonymized data
        if 'Anonymized' in path:
            generalized_a = data_dict['p']
            data_dict['p'] = sample_to_original_domain(data_dict['p'], shared_pre_calc_data['hierarchies'])
            data_dict['p'] = concat_generalized_columns(data_dict['p'], generalized_a, shared_pre_calc_data['qid'])

        spm = StructuredPrivacyMetrics(data_dict=data_dict,
                                       distance_dict=distance_dict,
                                       indices_dict=indices_dict,
                                       qid_distance_dict=distance_dict_qid,
                                       qid_indices_dict=indices_dict_qid,
                                       metadata=shared_pre_calc_data['meta_data'],
                                       hierarchies=shared_pre_calc_data['hierarchies'])

        # can be used to only calculate some of the metrics
        if distance_based_only or ml_only:
            privacy_metrics = dict()
            if distance_based_only:
                privacy_metrics.update(spm.get_distance_results())
            if ml_only:
                privacy_metrics.update(spm.get_ml_aia_metrics())
        else:
            privacy_metrics = spm.get_results()

        if not overwrite_old:
            # only overwrite calculated metrics and keep old
            if os.path.isfile(out_file):
                with open(out_file, 'r') as f:
                    old_results = json.load(f)
                old_results.update(privacy_metrics)
                privacy_metrics = old_results

        with open(out_file, 'w') as f:
            json.dump(privacy_metrics, f, cls=NpEncoder)

        if privacy_metrics["FAILED_RG"]:
            return f"{privacy_metrics['FAILED_RG']} failed for {path}"
        return None
    except Exception as e:
        raise Exception(f"Privacy failed for {path}") from e
        # return f"Privacy failed for {path}"


def calculate_crashed_priv(path):
    try:
        # keep same folder structures as the datasets
        out_file = path.replace('datasets', 'privacy').replace('.csv', '.json')
        os.makedirs('/'.join(out_file.split('/')[:-1]), exist_ok=True)

        original_path = f'{path.split("datasets")[0]}datasets/input/train.csv'
        holdout_path = f'{path.split("datasets")[0]}datasets/input/test.csv'

        o_data = pd.read_csv(original_path, dtype=str)
        h_data = pd.read_csv(holdout_path, dtype=str)
        p_data = pd.read_csv(path, dtype=str)

        data_dict = {
            'o': o_data,
            'p': p_data,
            'h': h_data
        }

        failed = True
        repeat_counter = 0
        while failed:
            repeat_counter += 1
            # all data must be in original domain so sampling trick must apply on anonymized data
            if 'Anonymized' in path:
                generalized_a = p_data
                data_dict['p'] = sample_to_original_domain(p_data, shared_pre_calc_data['hierarchies'])
                data_dict['p'] = concat_generalized_columns(data_dict['p'], generalized_a, shared_pre_calc_data['qid'])

            spm = StructuredPrivacyMetrics(data_dict=data_dict,
                                           distance_dict=None,
                                           indices_dict=None,
                                           qid_distance_dict=None,
                                           qid_indices_dict=None,
                                           metadata=shared_pre_calc_data['meta_data'],
                                           hierarchies=shared_pre_calc_data['hierarchies'])
            privacy_metrics = spm.get_gmm_results()

            if privacy_metrics["FAILED_RG"]:
                failed = True
            else:
                failed = False
            if repeat_counter == 10:
                raise Exception(f"Failed to calculate {path} after 10 tries")

        # read old results and update with new metrics
        with open(out_file, 'r') as f:
            old_privacy_metrics = json.load(f)
        old_privacy_metrics.update(privacy_metrics)
        with open(out_file, 'w') as f:
            json.dump(old_privacy_metrics, f, cls=NpEncoder)

    except Exception as e:
        raise Exception(f"Privacy failed for {path}") from e
        # return f"Privacy failed for {path}"


if __name__ == '__main__':
    # disables all multiprocessing
    debug_mode = False
    distance_based_only = False
    ml_only = False
    overwrite_old = False
    if debug_mode:
        print('MULTIPROCESSING IS DISABLED')
    # all jobs that use a lot of resources or is only done a limited amount of time will be done sequentially

    params = {'output_path': '../output/',
              'hierarchy_path': '../Hierarchies/',
              'test_ratio': 0.2,
              'print_all_anonymous': True,
              # false means that only 1 dataset will be created for each k-anon (for code testing)
              'synthetic_models': ['TVAE', 'CTGAN', 'diffusion'],
              'synth_model_obj': {
                  'TVAE': TVAE,
                  'CTGAN': CTGAN,
                  'diffusion': 'ddpm'
              },
              'synthetic_model_library': {
                  'TVAE': 'sdv',
                  'CTGAN': 'sdv',
                  'diffusion': 'synthcity'
              },
              'synth_samples': 5,
              'cat_metric': CatMetric.CUSTOM,
              'num_metric': NumMetric.CUSTOM,
              'use_weights': False,  # enables entropy based weights
              'n_neighbors': 5,
              'query_amount': 2000,
              'datasets_to_evaluate': ['ACSIncome', 'patients'],
              # remove elements here if you only want to run one of them
              'types_to_evaluate': ['original', 'synthetic', 'anonymous'],  # types of data to run utility/privacy on
              'dataset': {
                  'ACSIncome': {
                      'path': '../Datasets/ACSIncome.csv',
                      'synth_params': {
                          'synth_post_processor': None,
                          'synth_columns': ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'WKHP', 'SEX', 'RAC1P',
                                            'PINCP'],
                          'target': 'PINCP',
                          'TVAE': {
                              'batch_size': 1000
                          },
                          'CTGAN': {
                              'batch_size': 1000
                          },
                          'diffusion': dict(
                              is_classification=False,
                              n_iter=1000,  # epochs
                              lr=0.002,
                              weight_decay=1e-4,
                              batch_size=1000,
                              model_type="mlp",  # or "resnet"
                              model_params=dict(
                                  n_layers_hidden=8,
                                  n_units_hidden=256,
                                  dropout=0.0,
                              ),
                              num_timesteps=500,  # timesteps in diffusion
                              dim_embed=128
                          )
                      },
                      'columns': ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'WKHP', 'SEX', 'RAC1P', 'PINCP'],
                      'num': ['AGEP', 'WKHP', 'PINCP'],
                      'floats': ['WKHP', 'PINCP'],
                      'cat': ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'SEX', 'RAC1P'],
                      'qid': ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'SEX', 'RAC1P'],
                      'util_calculator': ACSIncome_util,
                      'util_params': {
                          'cores': 24,  # ml is very fast and no ram limits almost
                          'ml_types': [MLTYPE.BINARY_CLASSIFICATION, MLTYPE.REGRESSION,
                                       MLTYPE.MULTI_CLASS_CLASSIFICATION],
                          'generalize': True,
                          'drop_suppressed': False
                      },
                      'privacy_params': {
                          'cores': 16,
                          'sa': ['PINCP'],
                          'num_domains': {'AGEP': (1, 100)}
                          # must be pre given for numeric attributes that can be generalized,
                          # otherwise mistakes could be made when generalization ranges are larger than original domains
                      }
                  },
                  'patients': {
                      'path': '../Datasets/Patients_cleaned.csv',
                      'synth_params': {
                          'synth_post_processor': None,
                          'synth_columns': ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
                                            'bmi', 'smoking_status', 'stroke', 'Income', 'ZIP'],
                          'target': 'heart_disease',
                          'TVAE': {
                              'batch_size': 1000
                          },
                          'CTGAN': {
                              'batch_size': 1000
                          },
                          'diffusion': dict(
                              is_classification=True,
                              n_iter=1000,  # epochs
                              lr=0.002,
                              weight_decay=1e-4,
                              batch_size=1000,
                              model_type="mlp",  # or "resnet"
                              model_params=dict(
                                  n_layers_hidden=8,
                                  n_units_hidden=256,
                                  dropout=0.0,
                              ),
                              num_timesteps=500,  # timesteps in diffusion
                              dim_embed=128
                          )
                      },
                      'columns': ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                                  'smoking_status', 'stroke', 'Income', 'ZIP'],
                      'num': ['age', 'avg_glucose_level', 'bmi', 'Income'],
                      'floats': ['avg_glucose_level', 'bmi', 'Income'],
                      'cat': ['gender', 'hypertension', 'heart_disease', 'smoking_status', 'stroke', 'ZIP'],
                      'qid': ['gender', 'age', 'smoking_status', 'ZIP'],
                      'util_calculator': patients_util,
                      'util_params': {
                          'cores': 16,  # optimization is slower and uses 3GB each, pc has 128GB and 16 cores/32 threads
                          'sampling': True
                      },
                      'privacy_params': {
                          'cores': 16,
                          'sa': ['hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke', 'Income'],
                          'num_domains': {'age': (1, 100)}
                          # must be pre given for numeric attributes that can be generalized,
                          # otherwise mistakes could be made when generalization ranges are larger than original domains
                      }
                  }
              }}
    # comment out steps if something does not need to run
    pipeline_steps = [train_test_splits,
                      create_anonymized_datasets,
                      create_synthetic_datasets,
                      measure_utility,
                      measure_statistical_utility,
                      measure_privacy,
                      measure_crashed_privacy, # the likelihood metric can sometimes fail, this step reruns those
                      measure_additional_metrics]

    for step in tqdm(pipeline_steps, total=len(pipeline_steps), desc='Pipeline steps'):
        step(params)
