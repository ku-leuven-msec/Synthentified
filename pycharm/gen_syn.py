import time

import pandas as pd
import warnings

from sklearn.preprocessing import LabelEncoder
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

warnings.filterwarnings("ignore")

def generate_synthetic_sdv(o_data, num, gm, model_params: dict = None, f_path='syn_data', amount_of_samples=1,
                           cuda=False, post_processor=None):
    if model_params is None:
        model_params = {'cuda': cuda}
    else:
        model_params['cuda'] = cuda

    for i, n in enumerate(num):
        if num in list(o_data.columns):
            o_data.loc[:, n] = o_data[n].astype(float)

    discrete_columns = []
    # get discrete columns
    if 'object' in list(o_data.dtypes):
        discrete_columns = list(o_data.describe(include='object').keys())

    m = gm(**model_params)
    m.fit(o_data, discrete_columns)
    for i in range(amount_of_samples):
        # sample the same size as the original df
        s_data = m.sample(len(o_data))
        if post_processor is not None:
            s_data = post_processor(s_data)
        # transform dtype float to int for cols that always have .0 at the end
        for col in s_data.columns:
            if col in num and (s_data[col] % 1 == 0).all():
                s_data[col] = s_data[col].astype(int)
        s_data.to_csv(f'{f_path}/{gm.__name__.lower()}_{i}_syn.csv', index=False)


def generate_synthetic_synthcity(o_data: pd.DataFrame, num: list, gm: str, qid: list, target_column: str,
                                 model_params: dict = None, f_path='syn_data', amount_of_samples=1,
                                 post_processor=None):
    # Encode the data
    o_data_encoded = o_data.copy()
    categories_encoder_dict = dict()
    cat = [c for c in o_data.columns if c not in num]
    num = [n for n in num if num in list(o_data.columns)]

    for n in num:
        tmp = o_data_encoded[n].astype(float)
        if (tmp % 1 == 0).all():
            o_data_encoded.loc[:, n] = o_data_encoded[n].astype(int)
        else:
            o_data_encoded.loc[:, n] = tmp

    for v in cat:
        unique_vars = o_data_encoded[v].unique()
        categories_encoder_dict[v] = LabelEncoder()
        categories_encoder_dict[v].fit(unique_vars)
        o_data_encoded[v] = categories_encoder_dict[v].transform(o_data_encoded[v])

    # create the data loader object
    loader_encoded = GenericDataLoader(
        o_data_encoded,
        target_column=target_column,
        sensitive_columns=qid,
    )

    # get ddpm model
    plugin = Plugins().get(gm, **model_params)
    plugin.fit(loader_encoded)

    for i in range(amount_of_samples):
        # generate synthetic encoded data
        s_data = plugin.generate(count=len(o_data_encoded)).dataframe()

        # decode the synthetic data
        for v in cat:
            s_data[v] = categories_encoder_dict[v].inverse_transform(s_data[v])

        if post_processor is not None:
            s_data = post_processor(s_data)

        # transform dtype float to int for cols that always have .0 at the end
        for col in s_data.columns:
            if col in num and (s_data[col] % 1 == 0).all():
                s_data[col] = s_data[col].astype(int)
        s_data.to_csv(f'{f_path}/{gm}_{i}_syn.csv', index=False)
