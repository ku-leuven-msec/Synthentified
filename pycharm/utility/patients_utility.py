import numpy as np
import pandas as pd
import pyomo.environ as pyo
from haversine import Unit, haversine_vector
from pyomo.environ import ConcreteModel, Param, Var, RangeSet, Objective, Constraint, SolverFactory

import DistanceMetrics


def process_patients(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()

    def process_age(age):
        if isinstance(age, str):
            if age == "*":
                return 42
            elif '[' in age:
                age_bounds = age[1:len(age) - 1].split(", ")
                if not any(age_bounds):
                    return 42
                age_lb = float(age_bounds[0])
                age_ub = float(age_bounds[1])
                average = (age_lb + age_ub) / 2
                return int(average)
            else:
                return int(age)
        return age

    smoke_mapping = {
        'smokes': 1.0,
        'formerly smoked': 0.5,
        'ever smoked': 0.75,
        'never smoked': 0.5,
        '*': 0.5
    }

    df_processed["gender"] = df_processed["gender"].map(lambda entry: 1 if entry == "Male" else 0).astype(np.float32)
    df_processed["age"] = df_processed["age"].map(process_age).astype(np.float32)
    df_processed["hypertension"] = df_processed["hypertension"].astype(np.float32)
    df_processed["avg_glucose_level"] = df_processed["avg_glucose_level"].astype(np.float32)
    df_processed["bmi"] = df_processed["bmi"].astype(np.float32).replace(np.nan, 29.0)
    df_processed["smoking_status"] = df_processed["smoking_status"].map(smoke_mapping).replace(np.nan, 0.0)
    df_processed["Income"] = df_processed["Income"].astype(np.float32)

    df_processed["weight"] = (
                                     (0.05 * df_processed['gender']) +
                                     (0.15 * (df_processed['age'] - 1) / (82 - 1)) +
                                     (0.2 * df_processed['hypertension']) +
                                     (0.15 * (df_processed['avg_glucose_level'] - 55) / (281.59 - 55)) +
                                     (0.15 * (df_processed['bmi'] - 10.3) / (97.6 - 10.3)) +
                                     (0.1 * df_processed['smoking_status']) +
                                     (0.2 * df_processed['Income'] / 242857)
                             ) * 100

    return df_processed


def generate_model(df_patients, df_facilities, df_zipcodes, distances, nr_facilities_to_open=10):
    weights = df_patients["weight"]
    zipcodes_index_map = {a: idx for idx, a in enumerate(df_zipcodes["zip"])}
    patient_mapped_zipcode_indexes = df_patients["ZIP"].map(zipcodes_index_map)

    # optimization
    model = ConcreteModel()

    # Number of candidate locations
    model.m = Param(initialize=len(df_facilities))
    # Set of candidate locations
    model.M = RangeSet(1, len(df_facilities))
    # Set of customer nodes
    model.N = RangeSet(1, len(df_patients))

    # number of facilities
    model.p = Param(initialize=nr_facilities_to_open)
    # x[i,j] patient j that is supplied by resource i
    model.x = Var(model.M, model.N, within=pyo.Binary)
    # y[i] - a binary value that is 1 is a resource is located at location i
    model.y = Var(model.M, within=pyo.Binary)

    # objective function
    def cost_(models):
        return sum(
            weights[k - 1] * models.x[i, k] * distances[i - 1, patient_mapped_zipcode_indexes[k - 1]]
            for i in models.M for k in models.N)

    model.cost = Objective(rule=cost_)

    # patient j is allocated to exactly one resource
    def demand_(models, k):
        return sum(models.x[i, k] for i in models.M) == 1.0

    model.demand = Constraint(model.N, rule=demand_)

    # exactly p resource are located
    def facilities_(models):
        return sum(models.y[i] for i in models.M) == models.p

    model.facilities = Constraint(rule=facilities_)

    # patients can only be assigned to open resources
    def openfac_(models, i, k):
        return model.x[i, k] <= models.y[i]

    model.openfac = Constraint(model.M, model.N, rule=openfac_)

    return model


def optimize(model):
    # solver=glpk --solver-options="mipgap=0.02 cuts="
    solver = SolverFactory(
        'gurobi')  # here you can switch solvers (use GuRoBi if possible, but you have to have a license, otherwise use glpk)
    # solver.options['tmlim'] = 60
    solver.options['mipgap'] = 0.01
    solver.options['Threads'] = 2
    # solver.options['MipRelativeGap'] = 0.01

    result = solver.solve(model, tee=verbose_prints)
    return result


def calculate_cost_on_original(distances, chosen_facilities, df_patients, df_zipcodes):
    zipcodes_index_map = {a: idx for idx, a in enumerate(df_zipcodes["zip"])}
    patient_mapped_zipcode_indexes = df_patients["ZIP"].map(zipcodes_index_map)
    distances_2 = np.min(distances[chosen_facilities, :], axis=0)
    return np.sum(distances_2[patient_mapped_zipcode_indexes] * df_patients["weight"])


verbose_prints = False


def get_patients_utility(df_patients_test, df_facilities, df_patients_orig, df_zipcodes, verbose=False,
                         sampling=False):
    global verbose_prints
    verbose_prints = verbose

    facilities_to_open = 10

    df_patients_orig_processed = process_patients(df_patients_orig)

    if not sampling:
        # only zipcodes needed for zips in the datasets
        used_zips = np.unique(np.concatenate([df_patients_test['ZIP'].unique(), df_patients_orig['ZIP'].unique()]))
        df_zipcodes = df_zipcodes[df_zipcodes['zip'].isin(used_zips)]

        df_patients_test_processed = process_patients(df_patients_test)
        distances = haversine_vector(df_zipcodes[["latitude", "longitude"]], df_facilities[["Lat", "Lon"]],
                                     Unit.KILOMETERS,
                                     comb=True)
    else:
        # apply sampling trick
        qid = ['ZIP', 'age', 'smoking_status', 'gender']
        hierarchies = {qi: pd.read_csv(f'../Hierarchies/patients/{qi}.csv', sep=';', decimal=',',
                                       header=None, dtype=str) for qi in qid}

        df_patients_test = DistanceMetrics.sample_to_original_domain(df_patients_test, hierarchies)

        # only zipcodes needed for zips in the datasets
        used_zips = np.unique(np.concatenate([df_patients_test['ZIP'].unique(), df_patients_orig['ZIP'].unique()]))
        df_zipcodes = df_zipcodes[df_zipcodes['zip'].isin(used_zips)]

        distances = haversine_vector(df_zipcodes[["latitude", "longitude"]], df_facilities[["Lat", "Lon"]],
                                     Unit.KILOMETERS,
                                     comb=True)

        df_patients_test_processed = process_patients(df_patients_test)
    model = generate_model(df_patients_test_processed, df_facilities, df_zipcodes, distances, facilities_to_open)
    result = optimize(model)

    chosen_facilities = [i for i in range(0, len(df_facilities.Lon)) if model.y[i + 1].value > 0.0]

    cost_on_original = calculate_cost_on_original(distances, chosen_facilities, df_patients_orig_processed, df_zipcodes)

    return_items = {}
    return_items['model_score'] = pyo.value(model.cost)
    return_items['cost_on_original'] = cost_on_original
    return_items['facilities'] = chosen_facilities
    return return_items
