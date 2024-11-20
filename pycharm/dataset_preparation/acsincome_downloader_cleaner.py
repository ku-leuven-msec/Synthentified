from folktables import ACSDataSource, ACSIncome, generate_categories

year = '2018'
horizon = '1-Year'
survey = 'person'
sample_size = 33229
# set a fixed seed, we want our sample to stay the same one if we decide to change somthing in this cleanup code
seed = 0
output_path = '../../Datasets/ACSIncome.csv'

data_source = ACSDataSource(survey_year=year, horizon=horizon, survey=survey)
# overwrite the target transform to get the original income values
ACSIncome._target_transform = None
# downloads and filters for all states of america
data = data_source.get_data(download=True)
definition_df = data_source.get_definitions(download=True)
categories = generate_categories(features=ACSIncome.features, definition_df=definition_df)

ca_features, ca_labels, _ = ACSIncome.df_to_pandas(data, categories=categories, dummies=False)
# put target in same table
ca_features[ca_labels.columns[0]] = ca_labels.iloc[:, 0]

# WKHP and PINCP should also be int as all values end with .0
ca_features['WKHP'] = ca_features['WKHP'].astype(int)
ca_features['PINCP'] = ca_features['PINCP'].astype(int)

# create a sample (the full dataset is way too big to create synthetic datasets or distance metrics,
# additionally we want our results to not depend on this size so we set both patients and this dataset to be the same size
ca_features = ca_features.sample(n=int(sample_size), random_state=seed)

# drop RELP as the meaning of this columns isn't clear
ca_features.drop(columns=['RELP'], inplace=True)

ca_features.to_csv(f'{output_path}', sep=',', decimal='.', index=False)
