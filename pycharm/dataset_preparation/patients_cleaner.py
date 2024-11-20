import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../../Datasets/Patients.csv', sep=',', decimal='.', index_col='id', dtype=str)

    df['age'] = df['age'].astype(int)
    df['bmi'] = df['bmi'].astype(float)

    # fill in missing values assuming that individuals don't smoke before age 10
    df.loc[df['age'] < 10, 'smoking_status'] = 'never smoked'

    df['ageGrouped'] = np.round(df['age'], -1).astype(int)

    # interpolate smoke status and bmi
    smoketab = pd.crosstab([df['gender'], df['ageGrouped']], df['smoking_status']).apply(lambda x: x / x.sum(), axis=1)
    df.loc[df['smoking_status'].isna(), 'smoking_status'] = df[df['smoking_status'].isna()].apply(
        lambda x: smoketab.loc[x['gender']].loc[x['ageGrouped']].idxmax(), axis=1)

    bmiByAgeGender = df[~df['bmi'].isnull()].groupby(by=['gender', 'ageGrouped'])['bmi'].mean().round(1)
    df.loc[df['bmi'].isnull(), 'bmi'] = df[df['bmi'].isnull()].apply(
        lambda x: bmiByAgeGender.loc[x['gender']].loc[x['ageGrouped']], axis=1)

    df.drop(columns=['ageGrouped'], inplace=True)

    df.to_csv('../../Datasets/Patients_cleaned.csv', sep=',', decimal='.')
