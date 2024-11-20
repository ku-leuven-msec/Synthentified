import os.path
import re
import time
from enum import Enum
from functools import partial
from io import StringIO

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, \
    root_mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

import DistanceMetrics


class MLTYPE(Enum):
    # metric to use and target column filter
    REGRESSION = 0
    BINARY_CLASSIFICATION = 1
    MULTI_CLASS_CLASSIFICATION = 2


target_generators = {MLTYPE.REGRESSION: lambda df: df['PINCP'],
                     MLTYPE.BINARY_CLASSIFICATION: lambda df: df['PINCP'] > 50000,
                     MLTYPE.MULTI_CLASS_CLASSIFICATION: lambda df: pd.cut(df['PINCP'], bins=[0, 20000, 100000, np.inf],
                                                                          right=False, include_lowest=True).astype(str)}


def generalize_test_data(testing_set: pd.DataFrame, training_path: str, training_set: pd.DataFrame):
    # test if the given training_set is from within the anonymized folder by checking if a specific file exists
    if not os.path.isfile(f'{os.path.dirname(training_path)}/settings.csv'):
        return pd.read_csv(StringIO(testing_set.to_csv(index=False)))

    # retrieve QID from file
    qids = pd.read_csv(f'{os.path.dirname(training_path)}/settings.csv', sep=';', decimal='.').loc[0, 'qid']
    # transform string to list
    qids = qids[1:-1].split(', ')
    # retrieve generalization levels for each QID from the file name (if not available no generalization is assumed)
    levels = os.path.basename(training_path)[:-4].split('_')
    try:
        for i, level in enumerate(levels):
            levels[i] = int(level)
    except ValueError:
        levels = [0] * len(qids)

    for qid, level in zip(qids, levels):
        if level != 0:
            hierarchy = pd.read_csv(f'../Hierarchies/ACSIncome/{qid}.csv', sep=';', decimal='.', header=None,
                                    dtype=str)
            mapping = dict(zip(hierarchy[0].tolist(), hierarchy[level].tolist()))

            testing_set[qid] = testing_set[qid].map(mapping)

    # replace categorical values that where unknown in the training set with *
    for col in training_set.columns:
        if training_set[col].dtype == object:
            uniques = training_set[col].unique()
            testing_set.loc[~testing_set[col].isin(uniques), col] = '*'

    return testing_set


def sample_train_data(training_set: pd.DataFrame, training_path: str):
    # test if the given training_set is from within the anonymized folder by checking if a specific file exists
    if not os.path.isfile(f'{os.path.dirname(training_path)}/settings.csv'):
        return pd.read_csv(StringIO(training_set.to_csv(index=False)))

    # retrieve QID from file
    qids = pd.read_csv(f'{os.path.dirname(training_path)}/settings.csv', sep=';', decimal='.').loc[0, 'qid']
    qids = qids[1:-1].split(', ')

    hierarchies = {qid: pd.read_csv(f'../Hierarchies/ACSIncome/{qid}.csv', sep=';', decimal=',',
                                    header=None, dtype=str) for qid in qids}
    training_set = DistanceMetrics.sample_to_original_domain(training_set, hierarchies)

    return pd.read_csv(StringIO(training_set.to_csv(index=False)))


def get_pipeline(training_st: pd.DataFrame, ml_type: MLTYPE):
    pipe = None
    # guess what type each column is
    # filter numeric/categorical
    numerical, categorical = [], []
    for col in training_st.columns:
        if is_numeric_dtype(training_st[col]):
            numerical.append(col)
        else:
            # categorical should be one hot encoded
            categorical.append(col)

    if ml_type == MLTYPE.BINARY_CLASSIFICATION or ml_type == MLTYPE.MULTI_CLASS_CLASSIFICATION:
        pipe = Pipeline([
            ('preprocessing', ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical),
                    ("num", StandardScaler(), numerical)],
                remainder='passthrough',
                verbose=verbose_prints)),
            ('model', XGBClassifier(nthread=1))
        ], verbose=verbose_prints)
    if ml_type == MLTYPE.REGRESSION:
        pipe = Pipeline([
            ('preprocessing', ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical),
                    ("num", StandardScaler(), numerical)],
                remainder='passthrough',
                verbose=verbose_prints)),
            ('model', XGBRegressor(nthread=1))
        ], verbose=verbose_prints)
    return pipe


verbose_prints = False


def get_ml_utility(training_set: pd.DataFrame, testing_set: pd.DataFrame, ml_type: MLTYPE, training_path: str,
                   generalize: bool = False, drop_suppressed: bool = False, verbose=False):
    global verbose_prints
    verbose_prints = verbose

    training_set = training_set.copy()
    testing_set = testing_set.copy()

    # drop suppressed: this could improve the utility of anonymized datasets (what option makes the most sense?)
    if drop_suppressed:
        qid = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'SEX', 'RAC1P']
        suppressed_qids = ['*'] * len(qid)
        training_set.drop(training_set[(training_set[qid] == suppressed_qids).all(axis=1)].index, inplace=True)

    if generalize:
        def calculate_mean(range_str):
            numbers = re.findall(r'\d+', range_str)
            if len(numbers) == 2:
                start = int(numbers[0])
                end = int(numbers[1]) - 1
                return str((start + end) / 2)
            else:
                return range_str

        # replace AGEP generalizations with mean value
        must_replace = training_set['AGEP'].str.contains('\[|\*').any()
        if must_replace:
            training_set['AGEP'] = training_set['AGEP'].str.replace('*', '[1, 101[', regex=False)
            training_set['AGEP'] = training_set['AGEP'].apply(calculate_mean)

        # fix training dtypes
        training_set = pd.read_csv(StringIO(training_set.to_csv(index=False)))

        testing_set = generalize_test_data(testing_set, training_path, training_set)
        if must_replace:
            testing_set['AGEP'] = testing_set['AGEP'].str.replace('*', '[1, 101[', regex=False)
            testing_set['AGEP'] = testing_set['AGEP'].apply(calculate_mean)
        testing_set = pd.read_csv(StringIO(testing_set.to_csv(index=False)))

    else:
        training_set = sample_train_data(training_set, training_path)
        # fix testing dtypes
        testing_set = pd.read_csv(StringIO(testing_set.to_csv(index=False)))

    # fix the dtypes of train and test to be equal
    testing_set = testing_set.astype(training_set.dtypes.to_dict())

    learning_cols = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'WKHP', 'SEX', 'RAC1P']
    train_x, train_y = training_set[learning_cols], target_generators[ml_type](training_set)
    test_x, test_y = testing_set[learning_cols], target_generators[ml_type](testing_set)
    pipe = get_pipeline(train_x, ml_type)

    scores_dict = dict()
    if ml_type == MLTYPE.REGRESSION:
        pipe.fit(train_x, train_y)
        pred_y = pipe.predict(test_x)
        scores_dict['mse'] = mean_squared_error(test_y, pred_y)
        scores_dict['mae'] = mean_absolute_error(test_y, pred_y)
        scores_dict['rmse'] = root_mean_squared_error(test_y, pred_y)
        scores_dict['r2'] = r2_score(test_y, pred_y)
    else:
        # encode y values because xgboost does not follow the sklearn standard of being able to work with missing y values in classification
        # classes must also be numeric and this also fixes that
        classes_, y = np.unique(train_y, return_inverse=True)
        pipe.fit(train_x, y)
        pred_y = pipe.predict(test_x)
        pred_y = classes_[pred_y]
        report = classification_report(test_y, pred_y, output_dict=True)
        scores_dict = report['weighted avg']
        scores_dict['accuracy'] = report['accuracy']

    return scores_dict
