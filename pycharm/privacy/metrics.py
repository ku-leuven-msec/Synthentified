import copy
import itertools
import time
import warnings
from typing import Union

import networkx as nx
import numpy as np
from pgmpy.estimators import HillClimbSearch, K2Score
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd


class StructuredPrivacyMetrics:
    # NOTE: sample a_data before creating the StructuredPrivacyMetrics object
    def __init__(self,
                 data_dict: dict[str, pd.DataFrame],
                 distance_dict: dict[str, np.ndarray],
                 indices_dict: dict[str, np.ndarray],
                 qid_distance_dict: dict[str, np.ndarray],
                 qid_indices_dict: dict[str, np.ndarray],
                 metadata: dict[str, dict],
                 hierarchies: dict[str, pd.DataFrame]):
        self.distance_dict = distance_dict
        self.indices_dict = indices_dict

        self.qid_distance_dict = qid_distance_dict
        self.qid_indices_dict = qid_indices_dict

        self.metadata = metadata
        self.hierarchies = hierarchies

        self.sensitive_attributes = self.metadata['sa_names']
        self.num_attributes = self.metadata['num_names']
        self.cat_attributes = self.metadata['cat_names']
        self.other_attributes = self.metadata['other_names']
        self.qid_attributes = self.metadata['qid_names']
        self.attributes = self.metadata['columns']

        # This is only the mapping of first level generalization
        self.generalization_mapping = metadata['generalization_mapping']
        self.leaf_count_dict = metadata['leaf_count']
        self.categories_encoder_dict = metadata['categories_encoder_dict']
        self.leaves_of_gen = metadata['leaves_of_gen']

        self.all_columns_flags = [True, False]
        self.target_dataset_identifiers = ['h', 'o']  # h for holdout; o for original

        self.data_dict = \
            {key: self.safeguard_type(value, self.categories_encoder_dict) for key, value in data_dict.items()}

        self.std_dict = {'o': {sa: 0 for sa in self.sensitive_attributes},
                         'h': {sa: 0 for sa in self.sensitive_attributes},
                         'p': {sa: 0 for sa in self.sensitive_attributes}}

        # NOTE: the std dict is for calculating the approximate match and range match in the aia
        # NOTE: get std of the numerical data so that we can create range
        for key, item in self.data_dict.items():
            for attribute in self.num_attributes:
                self.std_dict[key][attribute] = item[attribute].std()

        self.generalized = False
        if len(data_dict['p'].columns) > len(data_dict['o'].columns):
            self.generalized = True

    def safeguard_type(self, data: pd.DataFrame, categories_encoder_dict: dict):
        temp_data = data.copy()
        for col_name in self.metadata['feature'].keys():
            if self.metadata['feature'][col_name]['value_type'] == 'categorical':
                encoder = categories_encoder_dict[col_name]
                temp_data[col_name] = encoder.transform(temp_data[col_name])
            else:
                temp_data[col_name] = temp_data[col_name].astype(float)
        return temp_data

    def get_results(self):

        result_dict = {**self.get_regular_metrics(),
                       **self.get_ml_aia_metrics()}

        return result_dict

    def get_gmm_results(self):
        dd_dict = dict()
        dd_dict["FAILED_RG"] = []
        for method in [self.__gmm_log_likelihood]:
            dd_dict[method.__name__[2:]] = dict()
            for flag in self.all_columns_flags:
                for tdi in self.target_dataset_identifiers:
                    # Note: Do not switch the order of these arguments
                    try:
                        dd_dict[method.__name__[2:]].update(method(flag, tdi))
                    except Exception as e:
                        if "FAILED_RG" not in dd_dict:
                            dd_dict["FAILED_RG"] = []
                        dd_dict["FAILED_RG"].append(f"{flag}-{tdi}-{method.__name__[2:]}")
        return dd_dict

    def get_distance_results(self):
        dd_dict = dict()
        dd_dict["FAILED_RG"] = []
        for method in [self.__nearest_neighbor_distance,
                       self.__close_value_ratio,
                       self.__similarity_ratio,
                       self.__nearest_neighbor_accuracy,
                       self.__nearest_neighbors_aia]:
            dd_dict[method.__name__[2:]] = dict()
            for flag in self.all_columns_flags:
                for tdi in self.target_dataset_identifiers:
                    # Note: Do not switch the order of these arguments
                    try:
                        dd_dict[method.__name__[2:]].update(method(flag, tdi))
                    except Exception as e:
                        if "FAILED_RG" not in dd_dict:
                            dd_dict["FAILED_RG"] = []
                        dd_dict["FAILED_RG"].append(f"{flag}-{tdi}-{method.__name__[2:]}")
        return dd_dict

    def get_regular_metrics(self):
        dd_dict = dict()
        dd_dict["FAILED_RG"] = []
        for method in [self.__nearest_neighbor_distance,
                       self.__close_value_ratio,
                       self.__similarity_ratio,
                       self.__nearest_neighbor_accuracy,
                       self.__gmm_log_likelihood,
                       self.__detection_rate,
                       self.__nearest_neighbors_aia]:
            dd_dict[method.__name__[2:]] = dict()
            for flag in self.all_columns_flags:
                for tdi in self.target_dataset_identifiers:
                    # Note: Do not switch the order of these arguments
                    try:
                        dd_dict[method.__name__[2:]].update(method(flag, tdi))
                    except Exception as e:
                        if "FAILED_RG" not in dd_dict:
                            dd_dict["FAILED_RG"] = []
                        dd_dict["FAILED_RG"].append(f"{flag}-{tdi}-{method.__name__[2:]}")

            dd_dict[self.relational_similarity.__name__] = dict()
            for flag in self.all_columns_flags:
                dd_dict[self.relational_similarity.__name__].update(self.relational_similarity(flag))
        return dd_dict

    def get_ml_aia_metrics(self):
        dd_dict = dict()
        ml_aia_methods = [self.qid_ml_aia,
                          self.all_ml_aia,
                          self.generalization_aia]
        # ml_aia_methods = []
        for method in ml_aia_methods:
            dd_dict[method.__name__] = method()
        return dd_dict

    def __calculate_distance(self, all_columns=True, target_set='o'):
        # the dimension of distance dict is (N, D)
        # where N is the number of samples
        # D is the predefined number of the nearest neighbors
        if all_columns:
            nearest_dists = np.array(self.distance_dict[f'p_{target_set}'][:, 0])
        else:
            nearest_dists = np.array(self.qid_distance_dict[f'p_{target_set}'][:, 0])

        return nearest_dists

    def __nearest_neighbor_distance(self, all_columns=True, target_set='o'):
        dists = self.__calculate_distance(all_columns, target_set)
        mean_dist = dists.mean()
        median_dist = np.median(dists)
        min_dist = np.min(dists)
        max_dist = np.max(dists)

        # Calculate the 10th and 90th percentiles
        percentile_10 = np.percentile(dists, 10)
        percentile_90 = np.percentile(dists, 90)

        # Filter values within the 10th to 90th percentile range
        filtered_dists = dists[(dists >= percentile_10) & (dists <= percentile_90)]
        mean_filtered_dist = filtered_dists.mean()
        median_filtered_dist = np.median(filtered_dists)
        min_filtered_dist = np.min(filtered_dists)
        max_filtered_dist = np.max(filtered_dists)

        return {f"nearest_neighbor_distance_mean_{target_set}_all_{all_columns}": mean_dist,
                f"nearest_neighbor_distance_median_{target_set}_all_{all_columns}": median_dist,
                f"nearest_neighbor_distance_min_{target_set}_all_{all_columns}": min_dist,
                f"nearest_neighbor_distance_max_{target_set}_all_{all_columns}": max_dist,
                f"nearest_neighbor_distance_mean_1090percentile_{target_set}_all_{all_columns}": mean_filtered_dist,
                f"nearest_neighbor_distance_median_1090percentile_{target_set}_all_{all_columns}": median_filtered_dist,
                f"nearest_neighbor_distance_min_1090percentile_{target_set}_all_{all_columns}": min_filtered_dist,
                f"nearest_neighbor_distance_max_1090percentile_{target_set}_all_{all_columns}": max_filtered_dist
                }

    def __close_value_ratio(self, all_columns=True, target_set='o'):
        dists = self.__calculate_distance(all_columns, target_set)
        result_dict = dict()
        close_value_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Calculate the 10th and 90th percentiles
        percentile_10 = np.percentile(dists, 10)
        percentile_90 = np.percentile(dists, 90)

        # Filter values within the 10th to 90th percentile range
        filtered_dists = dists[(dists >= percentile_10) & (dists <= percentile_90)]

        for cvt in close_value_thresholds:
            current_cvr = sum(dists < cvt) / len(dists)
            result_dict[f'close_value_ratio_{cvt}_{target_set}_all_{all_columns}'] = current_cvr

            filter_cvr = sum(filtered_dists < cvt) / len(filtered_dists)
            result_dict[f'close_value_ratio_1090percentile_{cvt}_{target_set}_all_{all_columns}'] = filter_cvr
        return result_dict

    def __similarity_ratio(self, all_columns=True, target_set='o'):
        distance_dict = self.distance_dict if all_columns else self.qid_distance_dict

        r = distance_dict[f'{target_set}_{target_set}'][:, 1]
        r_hat = distance_dict[f'p_{target_set}'][:, 0]

        n = len(r)
        # NOTE: according to the ADS-GAN paper, the greater the value the less the privacy
        sr = sum(r_hat < r) / n

        return {f"similarity_ratio_{target_set}_all_{all_columns}": sr}

    def __nearest_neighbor_accuracy(self, all_columns=True, target_set='o'):
        distance_dict = self.distance_dict if all_columns else self.qid_distance_dict

        r = distance_dict[f'{target_set}_{target_set}'][:, 1]
        r_hat = distance_dict[f'p_{target_set}'][:, 0]

        r_slash = distance_dict['p_p'][:, 1]
        r_slash_hat = distance_dict[f'{target_set}_p'][:, 0]

        # to avoid r and r_slash have different length
        comp1 = sum(r_hat > r) / (2 * len(r))
        comp2 = sum(r_slash_hat > r_slash) / (2 * len(r_slash))
        sr = comp1 + comp2
        return {f"nearest_neighbor_accuracy_{target_set}_all_{all_columns}": sr}

    def __gmm_log_likelihood(self, all_columns=True, target_set='o'):
        """
        Compute this metric.

        This fits multiple GaussianMixture models to the real data and then
        evaluates how likely it is that the synthetic data belongs to the same
        distribution as the real data.

        By default, GaussianMixture models will search for the optimal number of
        components and covariance type using the real data and then evaluate
        the likelihood of the synthetic data using those arguments 3 times.
        """
        if all_columns:
            real_data = self.data_dict[target_set].loc[:, self.num_attributes].values
            processed_data = self.data_dict['p'].loc[:, self.num_attributes].values
        else:
            qid_num_attributes = list(set(self.num_attributes).intersection(set(self.qid_attributes)))
            real_data = self.data_dict[target_set].loc[:, qid_num_attributes].values
            processed_data = self.data_dict['p'].loc[:, qid_num_attributes].values

        combinations = list(itertools.product(range(1, 31), ('diag',)))
        # select the best

        lowest_bic = np.inf
        best = None
        for n_components, covariance_type in combinations:
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
            try:
                gmm.fit(real_data)
                bic = gmm.bic(real_data)

                if bic < lowest_bic:
                    lowest_bic = bic
                    best = (n_components, covariance_type)
            except ValueError:
                pass
        if not best:
            # unable to fit GaussianMixture Model
            return None
        # Get the best combinations of n_components and covariance_type
        n_components, covariance_type = best
        scores = []
        # we hardcoded the iterations & retries
        iterations = 2
        retries = 2
        for _ in range(iterations * retries):
            try:
                gmm = GaussianMixture(n_components, covariance_type=covariance_type)
                gmm.fit(real_data)
                scores.append(gmm.score(processed_data))
                if len(scores) >= iterations:
                    break
            except ValueError:
                pass
        if not scores:
            # unable to compute the scores
            return None
        return {f"gmm_log_likelihood_{target_set}_all_{all_columns}": np.mean(scores)}

    def __detection_rate(self, all_columns=True, target_set='o'):

        if all_columns:
            o_data = self.data_dict[target_set]
            # to avoid involving the generalized attributes in the calculation
            p_data = self.data_dict['p'].loc[:, self.attributes]

        else:
            o_data = self.data_dict[target_set].loc[:, self.qid_attributes]
            p_data = self.data_dict['p'].loc[:, self.qid_attributes]

        Y_op = pd.Series([0] * len(p_data) + [1] * len(o_data), dtype=bool)
        # Vertically concatenate the DataFrames
        X_op = pd.concat([p_data, o_data])

        # feature type specification is used for xgb
        feature_types = list()
        for col_name in X_op.columns:
            if self.metadata['feature'][col_name]['value_type'] == 'numerical':
                feature_types.append("q")
            else:
                feature_types.append("c")

        def get_detection_result(clf, X_train, X_test, y_train, y_test):
            clf.fit(X_train, y_train)
            Y_pre = clf.predict(X_test)
            acc = sum(Y_pre == y_test) / len(y_test)
            return acc

        # NOTE: hard code the number of repetitions
        retries = 3
        xgb_results = list()

        # Suppress the convergence warning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        for _ in range(retries):
            X_train, X_test, y_train, y_test = train_test_split(X_op, Y_op, test_size=0.33)
            for _ in range(retries):
                # XGBClassifier
                xgb_results.append(get_detection_result(
                    XGBClassifier(n_estimators=30,
                                  feature_types=feature_types,
                                  max_depth=15,
                                  learning_rate=0.2,
                                  objective='binary:logistic',
                                  nthread=1),
                    X_train,
                    X_test,
                    y_train,
                    y_test))

        result_dict = {f"detection_rate_xgb_mean_{target_set}_all_{all_columns}": sum(xgb_results) / (retries ** 2),
                       f"detection_rate_xgb_max_{target_set}_all_{all_columns}": max(xgb_results),
                       f"detection_rate_xgb_min_{target_set}_all_{all_columns}": min(xgb_results),
                       }

        return result_dict

    def relational_similarity(self, all_columns=True):
        # NOTE: only fit the categorical values
        # evaluate this through Bayesian Network
        # convert the data into category only exluding some columns
        if all_columns:
            o_data = self.data_dict['o'].loc[:, self.cat_attributes]
            h_data = self.data_dict['h'].loc[:, self.cat_attributes]
            p_data = self.data_dict['p'].loc[:, self.cat_attributes]
            cat_attributes = list(set(self.cat_attributes) - set(self.other_attributes))
        else:
            qid_cat_attributes = list(set(self.cat_attributes).intersection(set(self.qid_attributes)))
            o_data = self.data_dict['o'].loc[:, qid_cat_attributes]
            h_data = self.data_dict['h'].loc[:, qid_cat_attributes]
            p_data = self.data_dict['p'].loc[:, qid_cat_attributes]
            cat_attributes = list(set(qid_cat_attributes) - set(self.other_attributes))
        # state names
        # We need to specify these
        # to avoid the original data, anonymized data and the synthetic data having different states
        # We use the states of the original data as the baselines
        state_names = {key: list(range(0, len(self.categories_encoder_dict[key].classes_))) for key in cat_attributes}
        estimated_model_dict = dict()

        # learn bayesian network model structure
        for kk, dd in {'o': o_data, 'h': h_data, 'p': p_data}.items():
            estimated_model_dict[kk] = HillClimbSearch(data=dd, state_names=state_names).estimate(
                scoring_method=K2Score(data=dd),
                max_indegree=None,
                max_iter=int(1e4),
                show_progress=False)

        # Function to evaluate the learned model structures.
        def get_f1_score(estimated_model, true_model):
            nodes = estimated_model.nodes()
            est_adj = nx.to_numpy_array(
                estimated_model.to_undirected(), nodelist=nodes, weight=None
            )
            true_adj = nx.to_numpy_array(
                true_model.to_undirected(), nodelist=nodes, weight=None
            )

            f1 = f1_score(np.ravel(true_adj), np.ravel(est_adj))

            return f1

        # Check the structure similarity of the bayesian network
        structure_similarity_op = get_f1_score(estimated_model_dict['p'], estimated_model_dict['o'])
        structure_similarity_hp = get_f1_score(estimated_model_dict['p'], estimated_model_dict['h'])
        return {f"relational_similarity_o_all_{all_columns}": structure_similarity_op,
                f"relational_similarity_h_all_{all_columns}": structure_similarity_hp}

    def __nearest_neighbors_aia(self, take_closest=True, target_set='o'):
        # NOTE: The attributes in the anonymized dataset and synthetic dataset can be categorized into
        #   1) quasi-identifiers
        #   2) sensitive attributes

        # NOTE: For each sensitive attribute, we assume that its information is missing.
        #   To recover the value of the sensitive attribute, from the attacker's perspective, what we can do is:
        #   1. Find one nearest neighbor according to the quasi_identifiers
        #   2. Get the value of the sensitive attribute of the closest neighbor as the inferred value
        #   3. Compared the inferred value with the ground truth and calculate the success rate

        sensitive_attributes = self.sensitive_attributes
        num_attributes = self.num_attributes
        len_data = len(self.qid_indices_dict[f'p_{target_set}'])

        postfix = f"{target_set}_closest_{take_closest}"
        temp_match_prob_dict = \
            {k: {sa: 0 for sa in sensitive_attributes} for k in ['match_error', 'range_overlap', 'guess_in_range']}
        match_prob_dict = \
            {f'{aiam}_{postfix}': dict() for aiam in ['match_error', 'range_overlap', 'guess_in_range']}

        for attribute in sensitive_attributes:
            for i in range(len_data):
                # we get its nearest neighbor index in the original data and find the corresponding data record
                p_record_index = self.qid_indices_dict[f'p_{target_set}'][i]
                # Get ground truth
                groundtruth_value = self.data_dict[target_set][attribute].iloc[i]
                # NOTE: If k == 1
                #   The guess value would be the value of the closest neighbor
                # Get guess value
                if take_closest:
                    guess_value = self.data_dict['p'][attribute][p_record_index].iloc[0]
                # NOTE: if k == 5 (5 is the max number of neighbors hard-coded in our code)
                else:
                    possible_guess_values = self.data_dict['p'][attribute][p_record_index].reset_index(drop=True)
                    # NOTE: if the current attribute is a numerical attribute
                    #   The guess value would be the average value of the closest neighbor
                    if attribute in num_attributes:
                        guess_value = sum(possible_guess_values) / len(possible_guess_values)
                    else:
                        guess_index = np.argmax(possible_guess_values)
                        guess_value = possible_guess_values.loc[guess_index]
                self.get_aia_accuracy(attribute, groundtruth_value, guess_value, temp_match_prob_dict)

            for aiam in ['match_error', 'range_overlap', 'guess_in_range']:
                match_prob_dict[f'{aiam}_{postfix}'][attribute] = temp_match_prob_dict[aiam][attribute] / len_data
        return match_prob_dict

    def fit_model(self,
                  attribute: str,
                  x_train: pd.DataFrame,
                  y_train: Union[pd.DataFrame, pd.Series],
                  x_test_dict: dict,
                  y_test_dict: dict,
                  all_columns: bool,
                  match_prob_dict: dict,
                  temp_match_prob_dict: dict,
                  normal_calculation: bool):
        # feature type specification is used for xgb
        feature_types = list()
        for col_name in x_train.columns:
            if self.metadata['feature'][col_name]['value_type'] == 'numerical':
                feature_types.append("q")
            else:
                feature_types.append("c")

        y_pred_dict = dict()
        if normal_calculation and self.metadata['feature'][attribute]['value_type'] == 'numerical':
            model = XGBRegressor(tree_method="hist",
                                 feature_types=feature_types,
                                 eval_metric=mean_absolute_error,
                                 nthread=1)
            model.fit(x_train, y_train)
            for key in x_test_dict.keys():
                y_pred_dict[key] = model.predict(x_test_dict[key])
        else:
            model = XGBClassifier(tree_method='hist',
                                  feature_types=feature_types,
                                  eval_metric='auc',
                                  nthread=1)
            # redo the class numbering on the training data
            original_values, new_values = np.unique(y_train, return_inverse=True)
            y_train = new_values
            model.fit(x_train, y_train)
            for key in x_test_dict.keys():
                y_pred_dict[key] = model.predict(x_test_dict[key])
                # convert the prediction to what the class numbering originally meant
                y_pred_dict[key] = original_values[y_pred_dict[key]]

        for key in x_test_dict.keys():
            len_data = len(y_test_dict[key])
            refreshed_match_prob_dict = copy.deepcopy(temp_match_prob_dict)

            # convert numeric ground truths back to strings such as in the hierarchy
            if not normal_calculation and y_test_dict[key].dtype == float and (y_test_dict[key] % 1 == 0).all():
                y_test_dict[key] = y_test_dict[key].astype(int).astype(str)

            for i in range(len_data):
                groundtruth_value = y_test_dict[key][i]
                guess_value = y_pred_dict[key][i]
                if normal_calculation:
                    self.get_aia_accuracy(attribute, groundtruth_value, guess_value, refreshed_match_prob_dict)
                else:
                    if self.metadata['feature'][attribute]['value_type'] != 'numerical' \
                            and self.metadata['feature'][attribute]['value_type'] != 'other':
                        groundtruth_value = \
                            self.categories_encoder_dict[attribute].inverse_transform([groundtruth_value])[0]
                        if not self.generalized:
                            guess_value = self.categories_encoder_dict[attribute].inverse_transform([guess_value])[0]
                    if not self.generalized and self.metadata['feature'][attribute]['value_type'] == 'numerical':
                        guess_value = str(int(guess_value))
                    self.get_generalized_aia_accuracy(attribute, groundtruth_value, guess_value, refreshed_match_prob_dict)
            postfix = f"{key}_all_{all_columns}"
            for aiam in refreshed_match_prob_dict.keys():
                match_prob_dict[f'{aiam}_{postfix}'][attribute] = refreshed_match_prob_dict[aiam][attribute] / len_data

    def qid_ml_aia(self):
        # NOTE: The attributes in the anonymized dataset and synthetic dataset can be categorized into
        #   1) quasi-identifiers
        #   2) sensitive attributes

        # NOTE: For each sensitive attribute, we assume that its information is missing.
        #   To recover the value of the sensitive attribute, from the attacker's perspective, what we can do is:
        #   1. Find one nearest neighbor according to the quasi_identifiers
        #   2. Get the value of the sensitive attribute of the closest neighbor as the inferred value
        #   3. Compared the inferred value with the ground truth and calculate the success rate

        # extract the subset of dataframe according to qid
        # NOTE: we are going to train on p and predict with o/h
        x_train = self.data_dict['p'].loc[:, self.qid_attributes]

        x_test_dict = dict()
        match_prob_dict = dict()
        for key in self.target_dataset_identifiers:
            x_test_dict[key] = self.data_dict[key].loc[:, self.qid_attributes]
            for aiam in ['match_error', 'range_overlap', 'guess_in_range']:
                match_prob_dict[f'{aiam}_{key}_all_{False}'] = dict()

        temp_match_prob_dict = {k: {sa: 0 for sa in self.sensitive_attributes} for k in
                                ['match_error', 'range_overlap', 'guess_in_range']}
        for attribute in self.sensitive_attributes:
            y_train = self.data_dict['p'][attribute]

            y_test_dict = dict()
            for key in self.target_dataset_identifiers:
                y_test_dict[key] = self.data_dict[key][attribute]

            self.fit_model(attribute,
                           x_train,
                           y_train,
                           x_test_dict,
                           y_test_dict,
                           False,
                           match_prob_dict,
                           temp_match_prob_dict,
                           normal_calculation=True)

        return match_prob_dict

    def all_ml_aia(self):
        try_out_attributes = list(set(self.data_dict['o'].columns) - set(self.other_attributes))
        temp_match_prob_dict = \
            {k: {toa: 0 for toa in try_out_attributes} for k in ['match_error', 'range_overlap', 'guess_in_range']}
        match_prob_dict = dict()
        for key in self.target_dataset_identifiers:
            for aiam in ['match_error', 'range_overlap', 'guess_in_range']:
                match_prob_dict[f'{aiam}_{key}_all_{True}'] = dict()

        for attribute in try_out_attributes:
            independent_vars = list(set(self.data_dict['o'].columns) - {attribute})
            x_train = self.data_dict['p'].loc[:, independent_vars]
            y_train = self.data_dict['p'][attribute]

            x_test_dict = dict()
            y_test_dict = dict()
            for key in self.target_dataset_identifiers:
                x_test_dict[key] = self.data_dict[key].loc[:, independent_vars]
                y_test_dict[key] = self.data_dict[key][attribute]

            self.fit_model(attribute,
                           x_train,
                           y_train,
                           x_test_dict,
                           y_test_dict,
                           True,
                           match_prob_dict,
                           temp_match_prob_dict,
                           normal_calculation=True)

        return match_prob_dict

    def generalization_aia(self):
        try_out_attributes = list(set(self.qid_attributes) - set(self.other_attributes))

        temp_match_prob_dict = \
            {k: {qid: 0 for qid in try_out_attributes} for k in ['match_error_generalization']}
        match_prob_dict = dict()

        for key in self.target_dataset_identifiers:
            match_prob_dict[f'match_error_generalization_{key}_all_{True}'] = dict()

        for qid in try_out_attributes:
            predictive_attributes = list(set(self.data_dict['o'].columns) - {qid})
            # the same name as the one defined in the helper
            if self.generalized:
                target_attribute = f'{qid}_generalized'
            else:
                target_attribute = f'{qid}'
            x_train = self.data_dict['p'].loc[:, predictive_attributes]
            y_train = self.data_dict['p'][target_attribute]

            x_test_dict = dict()
            y_test_dict = dict()
            for key in self.target_dataset_identifiers:
                x_test_dict[key] = self.data_dict[key].loc[:, predictive_attributes]
                y_test_dict[key] = self.data_dict[key][qid]

            self.fit_model(qid,
                           x_train,
                           y_train,
                           x_test_dict,
                           y_test_dict,
                           True,
                           match_prob_dict,
                           temp_match_prob_dict,
                           normal_calculation=False)

        return match_prob_dict

    def get_aia_accuracy(self,
                         attribute: str,
                         groundtruth_value: Union[int, float, str],
                         guess_value: Union[int, float, str],
                         match_prob_dict: dict):
        def is_overlap(original_attribute_range: tuple, processed_attribute_range: tuple):
            s1, e1 = original_attribute_range
            s2, e2 = processed_attribute_range
            #   Return the percentage of overlap
            #   The percentage of overlap is with respect to the original dataset
            interval_length_1 = e1 - s1
            interval_length_2 = e2 - s2

            if s1 <= s2 <= e1 <= e2:
                overlap = e1 - s2
            elif s1 <= s2 <= e2 <= e1:
                overlap = e2 - s2
            elif s2 <= s1 <= e1 <= e2:
                overlap = e1 - s1
            elif s2 <= s1 <= e2 <= e1:
                overlap = e2 - s1
            else:
                overlap = 0

            overlap_ratio = \
                overlap / interval_length_1 if interval_length_1 > interval_length_2 else overlap / interval_length_2

            return overlap_ratio

        if self.metadata['feature'][attribute]['value_type'] == 'numerical':
            groundtruth_std = self.std_dict['o'][attribute]
            groundtruth_range = \
                (max(0, groundtruth_value - groundtruth_std), groundtruth_value + groundtruth_std)

            guess_std = self.std_dict['p'][attribute]
            guess_range = (max(0, guess_value - guess_std), guess_value + guess_std)

            match_error = abs(guess_value - groundtruth_value)
            range_overlap = is_overlap(groundtruth_range, guess_range)
            guess_in_range = (groundtruth_range[0] <= guess_value <= groundtruth_range[1])
        else:
            groundtruth_range = self.generalization_mapping[attribute][groundtruth_value]
            guess_range = self.generalization_mapping[attribute][guess_value]

            match_error = (groundtruth_value == guess_value)
            range_overlap = 0
            # normalized by leaf
            guess_in_range = (groundtruth_range == guess_range) / (self.leaf_count_dict[attribute][groundtruth_value])

        match_prob_dict['match_error'][attribute] += match_error
        match_prob_dict['range_overlap'][attribute] += range_overlap
        match_prob_dict['guess_in_range'][attribute] += guess_in_range

    def get_generalized_aia_accuracy(self,
                                     attribute: str,
                                     groundtruth_value: Union[int, float, str],
                                     guess_value: Union[int, float, str],
                                     match_prob_dict: dict):


        # check if the ground truth is the descendant of the guess value
        descendant_bool = groundtruth_value in self.leaves_of_gen[attribute][guess_value]

        if descendant_bool:
            normalization_value = len(self.leaves_of_gen[attribute][guess_value])
            match_prob_dict['match_error_generalization'][attribute] += (1 / normalization_value)
        return None
