import pandas as pd
import numpy as np
from sklearn.datasets import load_digits


def load_adult_dataset():
    columns = ['age', 'workclass', 'fnlwgt', 'education',
               'education-num', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'capital-gain', 'capital-loss',
               'hours_per_week', 'native_country', 'target']

    df_hcm_data = pd.read_csv('adult.data', header=None)
    df_hcm_data.columns = columns

    categorical_cols = df_hcm_data.select_dtypes(include='object').columns.to_list()
    categorical_cols.remove('workclass')
    categorical_cols.remove('occupation')
    categorical_cols.remove('native_country')
    categorical_cols.remove('target')
    df_hcm_data = df_hcm_data.drop(['workclass', 'occupation', 'native_country'],
                                   axis=1)

    for col in categorical_cols:
        df_hcm_data[col] = df_hcm_data[col].apply(lambda x: x.strip())

    df_hcm_data_dummies = pd.get_dummies(df_hcm_data,
                                         columns=categorical_cols,
                                         drop_first=True)

    return df_hcm_data_dummies


def load_beans_dataset():
    df_beans = pd.read_excel('Dry_Bean_Dataset.xlsx')
    df_beans.rename(columns={'Class': 'target'}, inplace=True)
    return df_beans


def load_digits_dataset():
    X, y = load_digits(return_X_y=True)
    df_digits = pd.DataFrame(X)
    df_digits['target'] = y
    return df_digits


def load_chess_dataset():
    df_chess = pd.read_csv('chess.data', header=None)
    encoding = {'b': 0, 'f': 1, 'g': 2, 'l': 3, 'n': 4, 't': 5, 'w': 6}
    for col in list(range(36)):
        df_chess[col] = df_chess[col].apply(lambda x: encoding[x])
    df_chess = df_chess.rename(columns={36: 'target'})
    return df_chess


def load_diabetes_dataset():
    df_diabetes = pd.read_csv('messidor_features.csv', header=None)
    df_diabetes = df_diabetes.rename(columns={19: 'target'})
    return df_diabetes


def load_sensorless_dataset():
    df = pd.read_csv('Sensorless_drive_diagnosis.txt', sep=' ', header=None)
    df = df.rename(columns={48: 'target'})
    return df
