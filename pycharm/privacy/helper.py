import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder

from DistanceMetrics import DistanceBuilder, nearest_neighbors_from_matrix_heap
import os


def can_convert_to_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def calc_entropy(values):
    value, counts = np.unique(values, return_counts=True)
    return entropy(counts) + 1e-16


def create_leaf_count(hierarchy: pd.DataFrame):
    leaves_for_gen = dict()

    for level in hierarchy.columns:
        grouped = hierarchy.groupby(by=level)
        for group_name, df_group in grouped:
            leaves_for_gen[group_name] = df_group[0].tolist()

    if '*' not in leaves_for_gen:
        # add single * for suppression representing full domain
        leaves_for_gen['*'] = hierarchy[0].tolist()
    return leaves_for_gen


def concat_generalized_columns(a_sampled_data: pd.DataFrame, a_generalized_data: pd.DataFrame, qid: list):
    for q_name in qid:
        a_sampled_data[f'{q_name}_generalized'] = a_generalized_data[q_name]
    return a_sampled_data


def make_meta_data_for_privacy_eval(df: pd.DataFrame,
                                    run_params: dict,
                                    dataset_name,
                                    is_development=False):
    meta_data_dict = dict()

    # Note: df is the original data
    columns = run_params['dataset'][dataset_name]['columns']
    cat_names = run_params['dataset'][dataset_name]['cat']
    num_names = run_params['dataset'][dataset_name]['num']
    qid_names = run_params['dataset'][dataset_name]['qid']
    sa_names = run_params['dataset'][dataset_name]['privacy_params']['sa']
    hierarchy_path = run_params['hierarchy_path'] + dataset_name
    # sets overwrite values (needed when max gen range is larger than original data domain)
    n_domains = run_params['dataset'][dataset_name]['privacy_params']['num_domains']
    use_weights = run_params['use_weights']

    meta_data_dict['feature'] = dict()

    hierarchies_dict = dict()
    other_attributes_names = list()
    # NOTE: it stores the categorical encoder
    categories_encoder_dict = dict()
    mapping_dict = {c: dict() for c in df.columns}
    leaf_count_dict = {c: dict() for c in df.columns}
    leaves_of_gen = {c: dict() for c in qid_names}
    weights = dict() if use_weights else None

    for col in columns:
        unique_vars = df[col].unique()
        if col in cat_names:
            value_type = 'categorical'

            if use_weights:
                weights[col] = 1 / calc_entropy(df[col].values)

            # we record the categorical values in the original dataset to make sure the encoding is consistent
            # to avoid the following problem "What if the synthetic data has missed categories?"
            categories = unique_vars
            categories_encoder_dict[col] = LabelEncoder()
            categories_encoder_dict[col].fit(categories)
            if is_development and df[col].nunique() > 200:
                # IDEA: this has the same effect as marking 'other'
                #  we use this as a flag to exclude categorical attributes that have more than 30 categories
                #  in the distribution calculation as these result in extreem high calculation times
                values = None
                other_attributes_names.append(col)
            else:
                # ---------------------------------------------------------------------
                # ---------------------------------------------------------------------
                values = ','.join(unique_vars)
        else:
            value_type = 'numerical'
            float_col = df[col].astype(float)

            if use_weights:
                weights[col] = 1 / calc_entropy(np.round(float_col.values))

            if col not in n_domains:
                col_min_value = float_col.min()
                col_max_value = float_col.max()
                values = f'{col_min_value}, {col_max_value}'
                n_domains[col] = (col_min_value, col_max_value)
            else:
                values = f'{n_domains[col][0]}, {n_domains[col][1]}'
        if col in qid_names:
            privacy_type = 'qid'
        elif col in sa_names:
            privacy_type = 'sa'
        else:
            privacy_type = 'normal'

        file_path = f"{hierarchy_path}/{col}.csv"
        must_have_hierarchy_cols = list(set(cat_names).union(set(qid_names)))

        if col in must_have_hierarchy_cols:
            has_hierarchy = True
            if os.path.exists(file_path):
                hierarchies_dict[col] = pd.read_csv(file_path, header=None, sep=';', dtype=str)
            else:
                h = pd.DataFrame()
                h[0] = df[col].unique()
                h[1] = '*'
                hierarchies_dict[col] = h
            # ---------------------------------------------------------------------
            # ---------------------------------------------------------------------
            mapping_dict[col] = dict()
            leaf_count_dict[col] = dict()
            if value_type == 'categorical':
                # For each item in the mapping dict
                # 1. The key is the current encoded unique value
                # 2. The value is the first level generalization
                for v in unique_vars:
                    first_level = hierarchies_dict[col][1][hierarchies_dict[col][0] == v].iloc[0]
                    encoded_cat_value = categories_encoder_dict[col].transform([v])[0]
                    mapping_dict[col][encoded_cat_value] = first_level

                    leaf_count_dict[col][encoded_cat_value] = \
                        len(hierarchies_dict[col][0][hierarchies_dict[col][1] == first_level])

            leaves_of_gen[col] = create_leaf_count(hierarchies_dict[col])

        else:
            has_hierarchy = False
        meta_data_dict['feature'][col] = {'value_type': value_type,
                                          'values': values,
                                          'privacy_type': privacy_type,
                                          'has_hierarchy': has_hierarchy}

    # NOTE: initialize metadata dict
    meta_data_dict['columns'] = columns
    meta_data_dict['cat_names'] = cat_names
    meta_data_dict['num_names'] = num_names
    meta_data_dict['other_names'] = other_attributes_names
    meta_data_dict['qid_names'] = qid_names
    meta_data_dict['sa_names'] = sa_names
    meta_data_dict['categories_encoder_dict'] = categories_encoder_dict
    meta_data_dict['generalization_mapping'] = mapping_dict
    meta_data_dict['leaf_count'] = leaf_count_dict
    meta_data_dict['leaves_of_gen'] = leaves_of_gen

    # NOTE: the parameters for distance builder for all columns
    db_dict = {'hierarchies': hierarchies_dict,
               'cat': cat_names,
               'num': num_names,
               'cat_metric': run_params['cat_metric'],
               'num_metric': run_params['num_metric'],
               'num_domains': n_domains,
               'columns': columns,
               'weights': weights
               }

    return meta_data_dict, hierarchies_dict, db_dict


def get_distance_and_indices(data_dict, name_1, name_2, distance_metric_builder: DistanceBuilder,
                             bidirectional=True, qid=None, n_neighbors=5):
    if qid is None:
        qid = []

    names = [name_1] if name_1 == name_2 else [name_1, name_2]
    processed_data_dict = dict()
    for data_name in names:
        processed_data_dict[data_name] = distance_metric_builder.preprocess_dataset(data_dict[data_name])

    non_qid = [col for col in data_dict['o'].columns if col not in qid]

    distance_dict = dict()
    indices_dict = dict()

    distance_dict_qid = dict()
    indices_dict_qid = dict()

    distance_matrix = None

    if len(qid) != 0:
        # there are qid columns specified
        distance_matrix = distance_metric_builder.get_full_distance_matrix(processed_data_dict[name_1],
                                                                           processed_data_dict[name_2],
                                                                           dtype=np.float32, columns=qid)
        # calculate name_1-name_2 nearest neighbors on qid
        dis, ind = nearest_neighbors_from_matrix_heap(distance_matrix, n_neighbors, True)
        distance_dict_qid[f'{name_1}_{name_2}'] = dis
        indices_dict_qid[f'{name_1}_{name_2}'] = ind

        if bidirectional and len(names) != 1:
            # other direction is requested and makes sense
            dis, ind = nearest_neighbors_from_matrix_heap(distance_matrix.T, n_neighbors, True)
            distance_dict_qid[f'{name_2}_{name_1}'] = dis
            indices_dict_qid[f'{name_2}_{name_1}'] = ind

    if len(non_qid) != 0:
        # there are non_qid columns specified, calculate by possibly extending the previous matrix
        distance_matrix = distance_metric_builder.get_full_distance_matrix(processed_data_dict[name_1],
                                                                           processed_data_dict[name_2],
                                                                           dtype=np.float32, columns=non_qid,
                                                                           distance_matrix=distance_matrix,
                                                                           previous_column_count=len(qid))
        # calculate name_1-name_2 nearest neighbors on all columns
        dis, ind = nearest_neighbors_from_matrix_heap(distance_matrix, n_neighbors, True)
        distance_dict[f'{name_1}_{name_2}'] = dis
        indices_dict[f'{name_1}_{name_2}'] = ind

        if bidirectional and len(names) != 1:
            # other direction is requested and makes sense
            dis, ind = nearest_neighbors_from_matrix_heap(distance_matrix.T, n_neighbors, True)
            distance_dict[f'{name_2}_{name_1}'] = dis
            indices_dict[f'{name_2}_{name_1}'] = ind
    del distance_matrix

    return distance_dict, indices_dict, distance_dict_qid, indices_dict_qid
