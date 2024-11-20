import heapq
from enum import Enum

import numpy as np
import pandas as pd
from numba import njit


# The first element in the tuple specifies the type of data converter:
# - default: categories become multi-hot encoded
# - none: without doing any conversions

# The second element in the tuple is the distance function
class CatMetric(Enum):
    CUSTOM = ('none', 'custom_distance_cat')


class NumMetric(Enum):
    CUSTOM = ('default', 'custom_distance_num')


class CalculationMethod(Enum):
    USER_GIVEN = 0
    CUSTOM_CALCULATED = 1
    SIMPLE_NUMERIC = 2

# this is faster than sklearn
@njit
def nearest_neighbors_from_matrix_heap(dist, n_neighbors, euclidean=True):
    n = dist.shape[0]
    nearest_neighbors = np.empty((n, n_neighbors), dtype=np.int32)
    nearest_distances = np.empty((n, n_neighbors), dtype=np.float32)

    for i in range(n):
        row = dist[i]

        # Initialize the heap with the first k elements
        heap = [(-row[j], j) for j in range(n_neighbors)]
        heapq.heapify(heap)  # Convert list to a heap structure

        for j in range(n_neighbors, row.size):
            d = row[j]
            if d < -heap[0][0]:
                heapq.heappushpop(heap, (-d, j))

        # Extract indices and distances from the heap and sort them
        sorted_heap = sorted(heap, key=lambda x: -x[0])  # sort by distance in ascending order
        for idx in range(n_neighbors):
            nearest_distances[i, idx] = -sorted_heap[idx][0]
            nearest_neighbors[i, idx] = sorted_heap[idx][1]

    if euclidean:
        nearest_distances = np.sqrt(nearest_distances)

    return nearest_distances, nearest_neighbors


@njit
def add_indexed_elements(full_matrix, matrix, df_a_indices, df_b_indices):
    for i in range(df_a_indices.size):
        a_idx = df_a_indices[i]
        for j in range(df_b_indices.size):
            b_idx = df_b_indices[j]
            full_matrix[i, j] += matrix[a_idx, b_idx]
    return full_matrix


def sample_to_original_domain(df: pd.DataFrame, hierarchies: dict[str, pd.DataFrame]):
    # first create a mapping that can convert each generalization to its set of original attributes
    mapping = dict()
    for col_name, hierarchy in hierarchies.items():
        col_mapping = dict()
        for level in hierarchy.columns:
            grouped = hierarchy.groupby(by=level)
            for group_name, df_group in grouped:
                col_mapping[group_name] = df_group[0].tolist()
        mapping[col_name] = col_mapping

    for col_name in hierarchies.keys():
        if '*' not in mapping[col_name]:
            # add single * for suppression representing full domain
            mapping[col_name]['*'] = hierarchies[col_name][0].tolist()

    sampled_data = df.copy()
    for col in hierarchies.keys():
        original_col = df[col]
        counts = original_col.value_counts()
        for val, count in counts.items():
            sampled = np.random.choice(mapping[col][val], count)
            sampled_data.loc[original_col == val, col] = sampled
    return sampled_data


class DistanceBuilder:
    def _create_encodings(self, hierarchies) -> tuple[list[list[str]], list[dict[str, int]]]:
        # columns having hierarchies need a mapping for every value in the hierarchy
        encoders = {}
        reverse_encoders = {}
        for col_name, hierarchy in hierarchies.items():
            reverse_encoder = {}
            # append unique values in each column
            uniques = pd.melt(hierarchy)['value'].unique()
            encoder = uniques.tolist()
            for i, u in np.ndenumerate(uniques):
                reverse_encoder[u] = i[0]
            # add suppression encoding (not every hierarchy has the exact suppression value *
            if '*' not in encoder:
                encoder.append('*')
                reverse_encoder['*'] = len(encoder) - 1

            encoders[col_name] = encoder
            reverse_encoders[col_name] = reverse_encoder
        # convert to list on column index (should have faster access)
        encoders = [encoders[col_name] if col_name in encoders else None for col_index, col_name in
                    enumerate(self.columns)]
        reverse_encoders = [reverse_encoders[col_name] if col_name in reverse_encoders else None for col_index,
        col_name in enumerate(self.columns)]

        return encoders, reverse_encoders

    def _create_step_two_encoders(self, hierarchies, num_domains):
        # add conversions for generalized values, allowed string format in hierarchy are: strings representing a number,
        # strings representing a range, strings representing a masked number and other strings seen as categories
        # generate a dummy encoder when the metric does not use the default transformer, reduces memory and duration
        step_two_encoders = [None] * len(self.columns)
        for col_name, hierarchy in hierarchies.items():
            col_index = self.columns_to_index[col_name]
            # do encoding based on num or cat
            if col_name in self.cat:
                if self.cat_metric.value[0] == 'default':
                    hierarchy = hierarchy.copy()
                    reverse_encoder = self.reverse_encoders[col_index]
                    for l in hierarchy.columns:
                        hierarchy[l] = hierarchy[l].map(reverse_encoder)
                    step_two_encoder = np.zeros((len(reverse_encoder), len(hierarchy[0])), dtype=bool)
                    for level in hierarchy.columns:
                        unique_values = hierarchy[level].unique()
                        if level == 0:
                            for value in unique_values:
                                # first level so one-hot
                                step_two_encoder[value][value] = True
                        elif level != hierarchy.columns[-1]:
                            grouped = hierarchy.groupby(by=level)
                            for value in unique_values:
                                # calculate the original values that are generalized
                                g = grouped.get_group(value)[0].values
                                generalized_values = step_two_encoder[g]
                                # generalized_values = np.stack(grouped.get_group(value)[0].map(step_two_encoder))
                                base_value = np.logical_or.reduce(generalized_values).astype(int)
                                step_two_encoder[value] = base_value
                        else:
                            for value in unique_values:
                                # multi-hot encode
                                str_value = self.encoders[col_index][value]
                                if str_value == '*' or all(c == '*' for c in str_value):
                                    # suppression level in some way
                                    step_two_encoder[value] = np.ones(len(hierarchy[0]), dtype=bool)
                                    if str_value != '*':
                                        # add a copy for when '*' is not the only suppression value (can happen with masking)
                                        step_two_encoder[reverse_encoder['*']] = np.ones(len(hierarchy[0]), dtype=bool)
                else:
                    step_two_encoder = np.zeros(len(self.reverse_encoders[col_index]))
                step_two_encoders[col_index] = step_two_encoder.astype(np.int8)
            elif col_name in self.num:
                if self.num_metric.value[0] == 'default':
                    step_two_encoder = np.zeros(shape=(len(self.reverse_encoders[col_index]), 2), dtype=np.float32)
                    reverse_encoder = self.reverse_encoders[col_index]
                    for level in hierarchy.columns:
                        unique_values = hierarchy[level].unique()
                        for value in unique_values:
                            num_value = reverse_encoder[value]
                            # try decoding the value as if they represent number
                            # test if it is a single number
                            try:
                                f = float(value)
                                # a single number has itself as min and max value
                                step_two_encoder[num_value] = [f, f]
                            except ValueError:
                                if ', ' in value and '[' in value:
                                    # test if it is a range of numbers
                                    min_value, max_value = value[1:-1].split(', ')
                                    min_value, max_value = float(min_value), float(max_value)
                                    step_two_encoder[num_value] = [min_value, max_value - 1]
                                elif '*' == value[-1] and '*' != value[0]:
                                    # test if it is a masked number but not suppressed
                                    min_value, max_value = float(value.replace('*', '0')), float(
                                        value.replace('*', '9'))
                                    step_two_encoder[num_value] = [min_value, max_value]
                                elif value == '*' or all(c == '*' for c in value):
                                    # it is suppressed
                                    # numeric, suppression can be seen as generalized over full domain
                                    step_two_encoder[num_value] = num_domains[col_name]
                                    if value != '*':
                                        # add a copy for when '*' is not the only suppression value (can happen with masking)
                                        step_two_encoder[reverse_encoder['*']] = step_two_encoder[num_value]
                                else:
                                    raise ValueError(f'Cannot convert {value} to a numeric notation')
                else:
                    step_two_encoder = np.zeros(len(self.reverse_encoders[col_index]))
                step_two_encoders[col_index] = step_two_encoder
            else:
                continue
        return step_two_encoders

    def _format_hierarchies(self, hierarchies):
        children = dict()
        parents = dict()
        nodes_at_level = dict()
        for col_name, hierarchy in hierarchies.items():
            reverse_encoder = self.reverse_encoders[self.columns_to_index[col_name]]
            # assign for each level separately to fix a node occurring multiple times
            col_children = [None] * len(hierarchy.columns)
            col_parents = [None] * len(hierarchy.columns)
            col_nodes_at_level = [None] * len(hierarchy.columns)

            for level in hierarchy.columns:
                parents_set = False
                children_set = False
                nodes = hierarchy[level].unique()
                encoded = [reverse_encoder[node] for node in nodes]
                if level == hierarchy.columns[-1] and '*' not in nodes:
                    encoded.append(reverse_encoder['*'])
                col_nodes_at_level[level] = encoded
                col_children[level] = [None] * len(reverse_encoder)
                col_parents[level] = [None] * len(reverse_encoder)
                if level == 0:
                    # leafs have no children
                    for e in encoded:
                        col_children[level][e] = []
                    children_set = True
                if level == hierarchy.columns[-1]:
                    # top node has no parents
                    for e in encoded:
                        col_parents[level][e] = []
                    parents_set = True
                if not children_set or not parents_set:
                    grouped = hierarchy.groupby(by=level)
                    for group_name, df_group in grouped:
                        encode_name = reverse_encoder[group_name]
                        if not children_set:
                            # set al children of a node encoded
                            child_strings = df_group[level - 1].unique()
                            childs_encoded = [reverse_encoder[c] for c in child_strings]
                            col_children[level][encode_name] = childs_encoded
                            if level == hierarchy.columns[-1] and group_name != '*':
                                col_children[level][reverse_encoder['*']] = childs_encoded
                        if not parents_set:
                            # set the parent of each node (is only 1)
                            parent = [reverse_encoder[p] for p in df_group[level + 1].unique()]
                            if level + 1 == hierarchy.columns[-1] and reverse_encoder['*'] not in parent:
                                parent.append(reverse_encoder['*'])
                            col_parents[level][encode_name] = parent
            children[col_name] = col_children
            parents[col_name] = col_parents
            nodes_at_level[col_name] = col_nodes_at_level
        # convert to list on column index (should have faster access)
        children = [children[col_name] if col_name in children else None for col_index, col_name in
                    enumerate(self.columns)]
        parents = [parents[col_name] if col_name in parents else None for col_index, col_name in
                   enumerate(self.columns)]
        nodes_at_level = [nodes_at_level[col_name] if col_name in nodes_at_level else None for col_index, col_name in
                          enumerate(self.columns)]
        return children, parents, nodes_at_level


    def __init__(self,
                 hierarchies: dict[str, pd.DataFrame] = None,  # hierarchies of the anonymized dataset
                 columns: list[str] = None,  # all attribute names
                 cat: list[str] = None,  # all categorical attribute names
                 num: list[str] = None,  # all numerical attribute names
                 num_domains: dict[str, tuple[float, float]] = None,  # the ranges of numerical attributes
                 cat_metric: CatMetric = None,  # def. encoding and distance; element in the enum
                 num_metric: NumMetric = None,  # def. encoding and distance; element in the enum
                 weights: dict[str, float] = None):  # weights for each column in the distance
        """
        Initializes the builder by preparing multiple data transformations used in distance calculations.
        This can take some time. Do not make a new builder when exactly the same settings are used again.
        This class is not threadsafe when modifying any of the given settings.
        Create a new instance when you want to run this using different settings in different threats.
        """
        if hierarchies is None:
            hierarchies = dict()
        if columns is None:
            raise ValueError('Columns must be defined')
        if cat is None:
            cat = []
        if num is None:
            num = []
        if num_domains is None:
            num_domains = dict()
        if weights is None:
            weights = dict()

        # the column names in order of occurrence (needed because we will lose the column name once the data is an
        # array)
        self.columns = columns
        self.columns_to_index = {c: i for i, c in enumerate(self.columns)}
        # names of categorical/numeric columns (we can't auto-detect it because generalized number are also strings)
        self.cat = set(cat)
        self.num = set(num)

        for c in cat:
            if c not in hierarchies.keys():
                raise ValueError(
                    f'All categorical values must provide a hierarchy (may be single level), even ones that are not generalized: {c}')

        for n in num:
            if n not in num_domains.keys():
                raise ValueError(
                    f'All numeric values must provide a domain: {n}')

        # for each numeric column, what is the min and max value (aka domain) --> can be used in scaling distances
        self.num_domains = [num_domains[col_name] if col_name in num_domains else None for col_name in self.columns]
        self.cat_metric = cat_metric
        self.num_metric = num_metric

        if self.cat_metric == CatMetric.CUSTOM:
            for c in cat:
                if len(hierarchies[c].columns) < 2 or hierarchies[c].iloc[:, -1].nunique() != 1:
                    raise ValueError(
                        f'Categorical metric {self.cat_metric} requires a proper hierarchy for each categorical '
                        f'value. This is not satisfied for {c}.')

        # mapping of int to string in hierarchy and reverse
        self.encoders, self.reverse_encoders = self._create_encodings(hierarchies)
        # Stores mapping of string representations to usefull values:
        # ranges become tuples, masked ints become tuples and categorical values become multihot encodings
        self.step_two_encoders = self._create_step_two_encoders(hierarchies, num_domains)

        # stores for each hierarchy the
        self.children, self.parents, self.nodes_at_level = self._format_hierarchies(hierarchies)

        # transform cat and num to boolean lists on column index
        self.cat = [True if c in self.cat else False for c in self.columns]
        self.num = [True if n in self.num else False for n in self.columns]
        self.weights = [weights[col] if col in weights else 1.0 for col in self.columns]

    def preprocess_dataset(self, df: pd.DataFrame):
        df = df.copy()
        # filter and sort data by given columns
        df = df[self.columns]
        for index, col in enumerate(df.columns):
            if reverse_encoder := self.reverse_encoders[self.columns_to_index[col]]:
                df[col] = df[col].map(reverse_encoder)
            else:
                df[col] = df[col].astype(float)
        return df


    def get_full_distance_matrix(self, df_a, df_b, methods: dict[str, CalculationMethod] = None,
                                 matrices: dict[str, dict[str, dict[str, float]]] = None, dtype=np.float64,
                                 columns: list[str] = None, distance_matrix: np.array = None,
                                 previous_column_count: int = 0, sklearn_compatible: bool = False):
        """Returns the full distance matrix. Can be used in sklearn by setting metric='precomputed' in Sklearn.
        Can only be used when df_a and df_b are the same length. (sklearn wants it to be square for some reason)
        Uses at least (len(df_a)^2)*dtype memory at the end.
        Aggregates multiple columns on the fly in a Euclidean way resulting in max 2 times this memory usage during calculation.
        This memory can be cleared again after the sklearn NearestNeighbors has been created.
        Make sure when using this that the order of datasets being passed stays the same. Otherwise, you have to transpose the distance matrix.
        The columns list limits what columns should be used to calculate the distance (None==all).
        The distance_matrix can be used to pass a partial distance on a limited set of columns.
        The code wil add the distance of the specified columns to this existing distance. (reduces calculation time)
        Previous column count is used to denormalize the given distance_matrix.
        Distances are normalized between 0-1 for each column and in the end in total."""
        if sklearn_compatible and len(df_a) != len(df_b):
            raise ValueError('Both datasets must have the same size. This is a sklearn limitation.')

        if columns is None:
            columns = self.columns

        if previous_column_count <= 0 and distance_matrix is not None:
            raise ValueError('A given distance matrix cannot have been calculated on <=0 columns')

        must_calc_col = np.array([c in columns for c in self.columns])

        not_found = []
        for c in columns:
            if c not in self.columns:
                not_found.append(c)
        if len(not_found) != 0:
            print(
                f'The following columns where not specified during builder construction and wil be ignored: {not_found}')

        # detect method based on know properties: qid(data with hierarchy) are pre_calculated
        # a given matrix == user_given and other numbers are simple numeric
        methods_by_index = []
        for col_index in range(len(self.columns)):
            if methods and self.columns[col_index] in methods:
                methods_by_index.append(methods[self.columns[col_index]])
                continue
            if matrices and self.columns[col_index] in matrices:
                methods_by_index.append(CalculationMethod.USER_GIVEN)
                continue
            if self.num[col_index]:
                if self.step_two_encoders[col_index] is None:
                    methods_by_index.append(CalculationMethod.SIMPLE_NUMERIC)
                else:
                    methods_by_index.append(CalculationMethod.CUSTOM_CALCULATED)
            else:
                methods_by_index.append(CalculationMethod.CUSTOM_CALCULATED)

        if distance_matrix is None:
            full_matrix = np.zeros(shape=(len(df_a), len(df_b)), dtype=dtype)
        else:
            distance_matrix *= previous_column_count
            full_matrix = distance_matrix.T
            del distance_matrix

        for col_index, method in enumerate(methods_by_index):
            if not must_calc_col[col_index]:
                # skip non specified columns
                continue
            if method == CalculationMethod.SIMPLE_NUMERIC:
                # transform to uniques and indices
                df_a_uniques, df_a_indices = np.unique(df_a[df_a.columns[col_index]], return_inverse=True)
                df_b_uniques, df_b_indices = np.unique(df_b[df_b.columns[col_index]], return_inverse=True)
                # numeric distance, this is not symmetric as the uniques may differ
                matrix = self.__simple_numeric_distance_fast(df_a_uniques, df_b_uniques, col_index, dtype=dtype)
                # square values for aggregation
                matrix *= matrix
            else:
                if method == CalculationMethod.USER_GIVEN:
                    matrix, df_a_indices, df_b_indices = self.__user_given_method(df_a, df_b, matrices, col_index,
                                                                                  dtype=dtype)
                else:
                    matrix, df_a_indices, df_b_indices = self.__self_calculated_method(df_a, df_b, col_index,
                                                                                       dtype=dtype)
                # square values for aggregation
                matrix *= matrix
            matrix *= self.weights[col_index]

            full_matrix = add_indexed_elements(full_matrix, matrix, df_a_indices, df_b_indices)
            del matrix

        return full_matrix.T / (len(columns) + previous_column_count)


    def __user_given_method(self, df_a, df_b, matrices, col_index, dtype=np.float64):
        # this code has not been used or tested atm as we didn't need it after all
        # must transform the given dictionary to a full distance matrix
        # the dictionary is allowed to only include the distance in one direction, symmetry is assumed
        df_a, df_a_indices = np.unique(df_a, return_inverse=True)
        df_b, df_b_indices = np.unique(df_b, return_inverse=True)

        in_matrix = matrices[self.columns[col_index]]
        if encoder := self.encoders[col_index]:
            # we have an original to integer mapping
            matrix = np.zeros((len(df_a), len(df_b)), dtype=dtype)
            for index_a, df_a_val in enumerate(df_a):
                str_a = encoder[df_a_val]
                for index_b, df_b_val in enumerate(df_b):
                    str_b = encoder[df_b_val]
                    val = None
                    if str_a in in_matrix:
                        if str_b in in_matrix[str_a]:
                            val = in_matrix[str_a][str_b]
                    if val is None:
                        val = in_matrix[str_b][str_a]
                    matrix[index_a, index_b] = val

            return matrix, df_a_indices, df_b_indices

        elif (df_a[self.columns[col_index]] % 1 == 0).all() and (df_b[self.columns[col_index]] % 1 == 0).all():
            # the column has no mapping but is an integer on its own, use the given domain as bounds and scale appropriately
            matrix = np.zeros((len(df_a), len(df_b)), dtype=dtype)
            for index_a, df_a_val in enumerate(df_a):
                str_a = str(df_a_val)
                for index_b, df_b_val in enumerate(df_b):
                    str_b = str(df_a_val)
                    val = None
                    if str_a in in_matrix:
                        if str_b in in_matrix[str_a]:
                            val = in_matrix[str_a][str_b]
                    if val is None:
                        val = in_matrix[str_b][str_a]
                    matrix[index_a, index_b] = val

            return matrix, df_a_indices, df_b_indices
        else:
            # this is because we can't use floats as numpy array indices and dicts will be way slower
            raise ValueError(
                f'A user given matrix can only be used for integer values or values having a hierarchy: {self.columns[col_index]}')

    def __self_calculated_method(self, df_a, df_b, col_index, dtype=np.float64):
        __full_distance_metrics = {'custom_distance_num': self.__custom_distance_num_fast,
                                   'custom_distance_cat': self.__custom_distance_cat_fast}
        __full_data_converters = {'none': self.__none_data_converter_fast,
                                  'default': self.__default_data_converter_fast}

        # get what metric to use
        if self.num[col_index]:
            if self.step_two_encoders[col_index] is None:
                # numeric value that is never generalized
                if not ((df_a[self.columns[col_index]] % 1 == 0).all() and (
                        df_b[self.columns[col_index]] % 1 == 0).all()):
                    # the value are non integers, only on the fly calculation is possible
                    raise ValueError(f'Cannot pre-calculate column {self.columns[col_index]}.'
                                     f'Float values must provide a hierarchy (can be single level) in order to pre-calculate al combinations.')
                transformer = __full_data_converters['none']
                distance_metric = __full_distance_metrics['simple_numeric']
            else:
                transformer = __full_data_converters[self.num_metric.value[0]]
                distance_metric = __full_distance_metrics[self.num_metric.value[1]]
        elif self.cat[col_index]:
            # get categorical functions
            transformer = __full_data_converters[self.cat_metric.value[0]]
            distance_metric = __full_distance_metrics[self.cat_metric.value[1]]
        else:
            raise ValueError(f'Unknown how to calculate distance for {self.columns[col_index]}')

        x, y, x_indices, y_indices = transformer(df_a[self.columns[col_index]], df_b[self.columns[col_index]],
                                                 col_index)
        matrix = distance_metric(x, y, col_index, dtype)

        return matrix, x_indices, y_indices

    # faster transformer that only returns uniques and the respective inverse indices
    def __none_data_converter_fast(self, x, y, col_index):
        x_unique_ints, x_indices = np.unique(x.astype(int), return_inverse=True)
        y_unique_ints, y_indices = np.unique(y.astype(int), return_inverse=True)

        return x_unique_ints, y_unique_ints, x_indices, y_indices

    def __default_data_converter_fast(self, x, y, col_index):
        x_unique_ints, x_indices = np.unique(x.astype(int), return_inverse=True)
        y_unique_ints, y_indices = np.unique(y.astype(int), return_inverse=True)

        step_two_encoder_np = self.step_two_encoders[col_index]
        x_encoded = step_two_encoder_np[x_unique_ints]
        y_encoded = step_two_encoder_np[y_unique_ints]
        return x_encoded, y_encoded, x_indices, y_indices


    def __simple_numeric_distance_fast(self, x, y, col_index, dtype=np.float64):
        x = x.astype(dtype)
        y = y.astype(dtype)
        dom = self.num_domains[col_index]
        max_distance = dom[1] - dom[0]
        distances = np.abs(x[:, None] - y)
        return np.divide(distances, max_distance, dtype=dtype)


    def __hierarchycal_edge_distance_fast(self, x, y, col_index, dtype=np.float64):
        int_type = np.int64 if dtype == np.float64 else np.int32

        x = x.astype(int_type)
        y = y.astype(int_type)

        # Convert parents to a homogeneous NumPy array
        amount_of_nodes = len(self.parents[col_index][0])
        parents_array = np.full((len(self.parents[col_index]), amount_of_nodes), -1, dtype=int_type)
        for i, level in enumerate(self.parents[col_index]):
            parents_array[i, :len(level)] = [l[0] if l is not None and len(l) != 0 else -1 for l in level]

        # Get level of x and y in the hierarchy
        x_level = np.full(fill_value=-1, shape=x.shape, dtype=int_type)
        y_level = np.full(fill_value=-1, shape=y.shape, dtype=int_type)
        for level, nodes in enumerate(self.nodes_at_level[col_index]):
            x_level[(x_level == -1) & np.isin(x, nodes)] = level
            y_level[(y_level == -1) & np.isin(y, nodes)] = level

        # Initialize leave to root chain matrices
        max_level = len(self.parents[col_index])
        x_chain = np.full((len(x), max_level), -1, dtype=int_type)
        y_chain = np.full((len(y), max_level), -1, dtype=int_type)
        x_chain[np.arange(len(x)), x_level] = x
        y_chain[np.arange(len(y)), y_level] = y

        # set last level to '*'
        x_chain[:, -1] = self.reverse_encoders[col_index]['*']
        y_chain[:, -1] = self.reverse_encoders[col_index]['*']

        cur_x_level = np.min(x_level)
        cur_y_level = np.min(y_level)

        # Calculate chains
        for cur_x_l in range(cur_x_level, max_level - 2):
            # Get parent nodes
            p_x = parents_array[cur_x_l, x_chain[:, cur_x_l]]
            mask_x = p_x != -1
            x_chain[mask_x, cur_x_l + 1] = p_x[mask_x]

        for cur_y_l in range(cur_y_level, max_level - 2):
            # Get parent nodes
            p_y = parents_array[cur_y_l, y_chain[:, cur_y_l]]
            mask_y = p_y != -1
            y_chain[mask_y, cur_y_l + 1] = p_y[mask_y]

        # Find lowest matching node
        match_index = np.argmax((x_chain[:, np.newaxis, :] != -1) & (y_chain[np.newaxis, :, :] != -1) &
                                (x_chain[:, np.newaxis, :] == y_chain[np.newaxis, :, :]), axis=2,
                                out=np.zeros((len(x), len(y)), dtype=int_type))
        dev = dtype(len(self.nodes_at_level[col_index]) * 2 - 2)
        dev = 1 if dev == 0 else dev

        distance_matrix = np.divide((match_index - x_level[:, np.newaxis]) + (match_index - y_level), dev, dtype=dtype)

        return distance_matrix


    def __custom_distance_num_fast(self, x, y, col_index, dtype=np.float64):
        dom = self.num_domains[col_index]
        max_distance = dom[1] - dom[0]

        x_originals = x[:, 0] == x[:, 1]
        y_originals = y[:, 0] == y[:, 1]

        # filter which dataset has the generalizations and which the originals
        origs = x if np.all(x_originals) else y if np.all(y_originals) else None
        gens = x if np.any(~x_originals) else y if np.any(~y_originals) else None

        if origs is None:
            # there is no given dataset that has only original values, both are probably generalized
            return np.where(np.all(x[:, None] == y, axis=2), dtype(0), dtype(max_distance))
        if gens is not None:
            # filter for each original, generalized combination if one is the generalization of the other
            mask = (origs[:, 0][:, None] <= gens[:, 1]) & (origs[:, 0][:, None] >= gens[:, 0])

            if x is not origs:
                mask = mask.T

            # calculate values for each generalization
            tmp_gens = gens[np.any(mask, axis=0)].astype(int)
            # calculate range lengths
            lengths = tmp_gens[:, 1] - tmp_gens[:, 0] + 1
            # for each unique length calculate average manhattan between all values in the range using an easy formula
            unique_lengths = np.unique(lengths)
            distances = np.array([2 * sum(i * (N - i) for i in range(1, N + 1)) / (N ** 2) for N in unique_lengths])
            # assign to bigger distance matrix
            distance_matrix = np.zeros(shape=(len(origs), len(gens)), dtype=dtype)
            for l, d in zip(unique_lengths, distances):
                distance_matrix[:, np.where(lengths == l)] = d

            if x is not origs:
                distance_matrix = distance_matrix.T

            matrix = np.where(mask, distance_matrix, dtype(max_distance * max_distance))
        else:
            # distance between two original values is just the normal numeric distance
            x = x.astype(dtype)
            y = y.astype(dtype)
            num_distance = np.abs(x[x_originals, 0][:, None] - y[y_originals, 0])
            matrix = num_distance
        return np.divide(matrix, dtype(max_distance), dtype=dtype)

    def __custom_distance_cat_fast(self, x, y, col_index, dtype=np.float64):
        int_type = np.int64 if dtype == np.float64 else np.int32
        # test if one is generalized while the other isn't
        # if both are generalized return 0 when x==y and max otherwise
        # if both are original return edge distance
        # else return edge distance but replace distances where generalization does not match
        x_originals = np.isin(x, self.nodes_at_level[col_index][0])
        y_originals = np.isin(y, self.nodes_at_level[col_index][0])
        orig = x if np.all(x_originals) else y if np.all(y_originals) else None
        gen = x if np.any(~x_originals) else y if np.any(~y_originals) else None
        if orig is None:
            # there is no given dataset that has only original values, both are probably generalized
            return np.where(x[:, None] == y, dtype(0), dtype(len(self.parents[col_index]) * 2 - 2))
        edge_distances = self.__hierarchycal_edge_distance_fast(x, y, col_index, dtype=dtype)
        if gen is None:
            return edge_distances

        # first replace other versions of suppression with *
        gen = gen.copy()
        gen[np.isin(gen, self.nodes_at_level[col_index][-1])] = self.reverse_encoders[col_index]['*']
        # transform parents to numpy array for easier indexing
        amount_of_nodes = len(self.parents[col_index][0])
        parents_array = np.full((len(self.parents[col_index]), amount_of_nodes), -1, dtype=int_type)
        for i, level in enumerate(self.parents[col_index]):
            parents_array[i, :len(level)] = [l[0] if l is not None and len(l) != 0 else -1 for l in level]
        # replace last parent with *
        parents_array[-2][parents_array[-2] != -1] = self.reverse_encoders[col_index]['*']

        is_gen = np.zeros(shape=(len(orig), len(gen)), dtype=bool)
        # gradually generalized the original value, mark each orig, gen combination where the generalization matches one in gen as possible gen
        orig_gen = orig.copy()
        for l in range(len(self.parents[col_index])):
            is_gen = np.logical_or(orig_gen[:, None] == gen, is_gen)
            orig_gen = parents_array[l][orig_gen]
        max_distance = (len(self.parents[col_index]) * 2 - 2)

        if orig is not x:
            is_gen = is_gen.T

        distances = np.where(is_gen, edge_distances, dtype(max_distance))
        return distances
