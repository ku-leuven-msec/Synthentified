from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# NOTE: replacing al categorical values with ints is way faster so DO THIS
#  accuracy of queries is also a bit flawed as it's accuracy is heavily correlated to the assumed distribution accuracy in comparison with the real distribution
#  this is why some papers put boundaries on queries where an eq is full or not included at all (this give very wide bound and not a single value estimate
#  see: "Aggregate Query Answering on Anonymized Tables"
#  other possibility is instead of estimating it using exact distributions (how I did it in previous paper), apply sampling trick, this would make it sample dependent but a lot easier to measure, we also have this problem with similarity measures

def get_uniform_probabilities(hierarchies: dict[str, pd.DataFrame], encoders: dict[str, LabelEncoder]):
    # atm we use numpy arrays and in quite a stupid way
    # (it is a square matrix while a rectangle of (nodes,originals) would work this needs our own label transformer)
    contains_dict = dict()
    probability_dict = dict()

    for col, hierarchy in hierarchies.items():

        # add '*' to hierarchy when not present
        if hierarchy.iloc[0, -1] != '*':
            hierarchy[hierarchy.columns[-1] + 1] = '*'

        node_count = len(np.unique(hierarchy))

        contains_array = np.zeros((node_count, node_count), dtype=bool)
        probability_array = np.zeros((node_count, node_count), dtype=np.float32)

        # transform hierarchy to numeric representation
        for level in hierarchy.columns:
            hierarchy[level] = encoders[col].transform(hierarchy[level])

        # add level 0, done separately because grouping this is heavier and stupid
        originals = hierarchy[0]
        contains_array[originals, originals] = True
        probability_array[originals, originals] = 1

        for level in range(1, len(hierarchy.columns)):
            grouped = hierarchy.groupby(by=level)
            for gen, group in grouped:
                originals = group[0]
                contains_array[gen, originals] = True
                probability_array[gen, originals] = 1 / len(originals)
        contains_dict[col] = contains_array
        probability_dict[col] = probability_array

    return contains_dict, probability_dict


def generate_random_queries(amount, columns, cat_options: dict[str, np.array],
                            pre_defined_filters: dict[str, list[Callable]] = None):
    """
    generates random set of queries. The query is an isin check for a random set out of cat_options.
    pre_defined_filters is a list of lambda's for a column. The isin query will be replaced by this lambda.
    A random lambda will be selected for the query. If you want full manual queries you will need to add them yourself.
    Queries are always of the form: count(x) where filter1 AND filter2 AND... where a filter is on a single column.
    More complex queries need multiple queries that are joined somehow or more complex evaluation code
    The isin query type could be done in a faster way, this will however remove the option of pre_defined_filters on qid
    """

    if pre_defined_filters is None:
        pre_defined_filters = dict()

    queries = []

    rng = np.random.default_rng(69)

    for _ in range(amount):
        # randomize how many columns to query over
        column_amount = rng.integers(1, len(columns), endpoint=True)
        selected_columns = rng.choice(len(columns), size=column_amount, replace=False, shuffle=False)

        joined_query = []

        for col in selected_columns:
            col_name = columns[col]
            if columns[col] in pre_defined_filters:
                possible_filters = pre_defined_filters[col_name]
                # select random pre_defined_filter
                filter_index = rng.integers(0, len(possible_filters))
                col_query = possible_filters[filter_index]
            else:
                # select random subset for isin query
                cat_set = cat_options[col_name]
                subset_size = rng.integers(1, len(cat_set), endpoint=True)
                cat_subset = rng.choice(cat_set, size=subset_size, replace=False, shuffle=False)
                col_query = lambda x, cat_subset_tmp=cat_subset: np.isin(x, cat_subset_tmp)
            joined_query.append(col_query)
        queries.append([selected_columns, joined_query])

    return queries


def evaluate_queries_original_domain(df, queries):
    """assumes df is in original domain, partitioning and encoding must have been applied already"""
    df = df.values
    rows = len(df)
    df = df.T
    counts = []

    for query in queries:
        columns = query[0]
        filters = query[1]

        filtered_data = df[columns]
        final_filter = np.ones(rows, dtype=bool)
        for col in range(len(columns)):
            final_filter &= filters[col](filtered_data[col])
        counts.append(final_filter.sum())
    return np.array(counts)


def estimate_queries(df, contains_dict, probability_dict, queries, qid, columns):
    # map qid to column indices
    df = df.values
    qid = [i for i, c in enumerate(columns) if c in qid]
    counts = []

    for query in queries:
        query_cols = query[0]
        filters = query[1]
        result = 0

        # a cache storing filter results of this query for a specific column value
        cache = {col: np.full(len(p_dict), -1, dtype=float) for col, p_dict in probability_dict.items()}

        # first filter total dataset
        filtered_df = df[:, query_cols]
        # first apply filters of columns not in qid
        for i, col in enumerate(query_cols):
            if col not in qid:
                filtered_df = filtered_df[filters[i](filtered_df[:, i])]
                if len(filtered_df) == 0:
                    break
        else:
            # when there is data left group and apply other part of queries
            qid_in_query = [i for i, col in enumerate(query_cols) if col in qid]

            if len(qid_in_query) == 0:
                result += len(filtered_df)
                counts.append(result)
                continue

            # only need unique row and group size, meaning this can be optimized further if needed
            qid_df = filtered_df[:, qid_in_query]
            eq_values, eq_sizes = np.unique(qid_df, axis=0, return_counts=True)

            # for group in grouped:
            for eq_value, eq_size in zip(eq_values, eq_sizes):
                current_prob = 1.0
                for i, j in enumerate(qid_in_query):
                    col = query_cols[j]
                    if (prob := cache[columns[col]][eq_value[i]]) != -1:
                        if prob == 0:
                            break
                    else:
                        # first get values in generalization
                        values = np.where(contains_dict[columns[col]][eq_value[i]])[0]
                        filtered_values = values[filters[j](values)]
                        if len(filtered_values) == 0:
                            # store result for this cols eq_value in a cache
                            cache[columns[col]][eq_value[i]] = 0
                            break
                        prob = np.sum(probability_dict[columns[col]][eq_value[i]][filtered_values])
                        # store result for this cols eq_value in a cache
                        cache[columns[col]][eq_value[i]] = prob

                    current_prob *= prob
                else:
                    # if all columns queried still return data (aka both for loops didn't break)
                    result += eq_size * current_prob
        counts.append(result)
    return counts


def partition_num_data(df, partitions: dict[str, list[int]] | dict[str, int]):
    """transform numeric floats to partitions. User given partitions or uniform split into given amount."""
    df_c = df.copy()
    for col, part in partitions.items():
        df_c[col] = pd.cut(df_c[col].astype(float), part, right=False, include_lowest=True).astype(str)

    if isinstance(list(partitions.items())[0][1], int):
        # ints where given, calculate created bucket boundaries to return
        buckets = dict()
        for col, part in partitions.items():
            buckets[col] = pd.cut(df[col].astype(float), part, right=False, include_lowest=True, retbins=True)[1]
    else:
        buckets = partitions
    return df_c, buckets


def get_encoders(df, hierarchies, excludes=None):
    """creates for each column an encoder, assumes that numerical columns are already binned or excluded.
    returns encoders and original cat_options"""

    if excludes is None:
        excludes = set()

    encoders = dict()
    cat_options = dict()
    for col in df.columns:
        if col in excludes:
            continue
        if col in hierarchies:
            uniques = np.unique(hierarchies[col])
            originals = hierarchies[col][0].values
        else:
            uniques = np.unique(df[col])
            originals = uniques
        # add '*' when not present
        if '*' not in uniques:
            uniques = np.append(uniques, '*')
        encoders[col] = LabelEncoder().fit(uniques)
        cat_options[col] = encoders[col].transform(originals)
    return encoders, cat_options


def get_encoded(df, encoders: dict[str, LabelEncoder]):
    df = df.copy()
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    # data should al be numerical by now, we return np array, and assume that the column order never changes
    return df
