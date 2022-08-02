import numpy as np
import pandas as pd
import time
from deepforest import CascadeForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
import tensorflow as tf


def get_sample(df, size, seed):
    """
    Returns a data set sample from the given size
    while maintaining class proportions.
    """
    df_size = len(df)
    class_counts = df['target'].value_counts()
    percentages = (size + 1) * (class_counts / df_size)
    df_sample = df.groupby('target', group_keys=False).apply(lambda x:
                                                             x.sample(n=int(percentages[x['target'].iloc[0]]),
                                                                      random_state=seed))
    return df_sample


def split_train_test(df):
    """
    Splits data with train (70%) and test (30%).
    y is assigned to column 'target' and x to any other feature.
    y is encoded using a LabelEncoder.
    """
    x = df.loc[:, df.columns != 'target']
    y = df.loc[:, 'target']
    x = x.astype('float32')
    y = LabelEncoder().fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return x_train, x_test, y_train, y_test


def train_deep_forest(x_train, y_train):
    """
    Returns a trained Deep Forest with the given data.
    There is a bug in the code for Deep Forest and the
    classifier can't be reused in a loop.
    """
    clf = CascadeForestClassifier(random_state=0, verbose=0)
    clf.fit(x_train, y_train)
    return clf


def train_neural_network(x_train, y_train):
    """
    Returns a trained Neural Network with the given data.
    There is a distinction between binary or multiclass case
    but the Neural Network architecture is the same.
    """
    n_classes = len(np.unique(y_train))
    n_features = x_train.shape[1]
    sequential_list = [InputLayer(n_features),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal()),
                          Dropout(0.1),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal()),
                          Dropout(0.1),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal()),
                          Dropout(0.1),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal()),
                          Dropout(0.1),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal()),
                          Dropout(0.1),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal()),
                          Dropout(0.1),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal()),
                          Dropout(0.1),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal()),
                          Dropout(0.1),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal()),
                          Dropout(0.1),
                          Dense(100, activation = "elu",kernel_initializer=tf.keras.initializers.HeNormal())
                      ]
    if n_classes > 2:
        sequential_list.append(Dense(n_classes + 1, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
    else:
        sequential_list.append(Dense(1, activation="sigmoid"))
        loss = 'binary_crossentropy'

    clf = Sequential(sequential_list)
    clf.compile(loss=loss,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
    clf.fit(x_train, y_train, epochs=200, batch_size=32, verbose=0)

    return clf


def compute_accuracy(x, y, clf, model):
    """
    Computes accuracy of the model for the given X and y.
    Returns model accuracy and time that takes to make
    each prediction
    """
    start_time = time.time()
    prediction = clf.predict(x).flatten()
    end_time = time.time()

    time_prediction = end_time - start_time

    if model == 'DNN':
        loss, accuracy = clf.evaluate(x, y, verbose=0)

    else:
        hits = sum(prediction == y)
        total = len(y)
        accuracy = hits / total

    accuracy = 100 * round(accuracy, 2)

    return {'accuracy': accuracy, 'time': time_prediction}


def get_results(df, seeds, model):
    """
    Returns a dictionary with multiple dataframes with
    accuracy for train set, accuracy for test set,
    time for training, time to do a prediction for train set,
    time to do a prediction for test set for all iterations
    and seeds.
    """
    size = 1000
    total_iters = 10
    cols = range(total_iters)
    scaler = MinMaxScaler()

    results = {'accuracy_train': pd.DataFrame(columns=cols),
               'accuracy_test': pd.DataFrame(columns=cols),
               'time_training': pd.DataFrame(columns=cols),
               'time_prediction_train': pd.DataFrame(columns=cols),
               'time_prediction_test': pd.DataFrame(columns=cols)}

    for seed in seeds:

        if (seed + 1) % 5 == 0:
            print(f'\n Calculating seed {seed + 1} out of {len(seeds)}')
            print('-------------------------')

        df_sample = get_sample(df, 1000, seed)
        df_remaining_data = df_sample.copy()
        size_sample = int(size / 10)

        df_iter = pd.DataFrame()
        accuracies_train, accuracies_test = [], []
        times_train, times_prediction_train, times_prediction_test = [], [], []

        for n_iter in range(total_iters):

            df_new_sample = get_sample(df_remaining_data, size_sample, 42)
            df_iter = pd.concat([df_iter, df_new_sample])
            df_remaining_data = df_remaining_data[~df_remaining_data.index.isin(df_iter.index)]

            x_train, x_test, y_train, y_test = split_train_test(df_iter)
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            start_time = time.time()

            if model == 'DF':
                clf = train_deep_forest(x_train, y_train)

            if model == 'DNN':
                clf = train_neural_network(x_train, y_train)

            if model == 'RF':
                clf = RandomForestClassifier(random_state=42)
                clf.fit(x_train, y_train)

            if model == 'DT':
                clf = DecisionTreeClassifier(random_state=42)
                clf.fit(x_train, y_train)

            if model == 'SVM':
                clf = svm.SVC(random_state=42)
                clf.fit(x_train, y_train)

            end_time = time.time()

            training_time = end_time - start_time

            results_train = compute_accuracy(x_train, y_train, clf, model)
            results_test = compute_accuracy(x_test, y_test, clf, model)

            accuracies_train.append(results_train['accuracy'])
            accuracies_test.append(results_test['accuracy'])
            times_prediction_train.append(results_train['time'])
            times_prediction_test.append(results_test['time'])
            times_train.append(training_time)

        results['accuracy_train'].loc[seed] = accuracies_train
        results['accuracy_test'].loc[seed] = accuracies_test
        results['time_training'].loc[seed] = times_train
        results['time_prediction_train'].loc[seed] = times_prediction_train
        results['time_prediction_test'].loc[seed] = times_prediction_test

    return results


def compute_all_models_results(df, n_seeds=30):
    """
    Computes the results for all the models
    """
    seeds = list(range(n_seeds))

    print('Starting iterations with DNN....')
    start_time = time.time()
    results_neural_network = get_results(df,seeds,'DNN')
    end_time  = time.time()
    print(f'\n Execution time for DNN is {end_time - start_time}')
    print('--------------------------------------')

    print('Starting iterations with DF....')
    start_time = time.time()
    results_deep_forest = get_results(df, seeds,'DF')
    end_time = time.time()
    print(f'\n Execution time for DF is {end_time - start_time}')
    print('--------------------------------------')

    print('Starting iterations with RF....')
    start_time = time.time()
    results_random_forest = get_results(df, seeds, 'RF')
    end_time = time.time()
    print(f'\n Execution time for RF is {end_time - start_time}')
    print('--------------------------------------')

    print('Starting iterations with DT....')
    start_time = time.time()
    results_decision_tree = get_results(df, seeds, 'DT')
    end_time = time.time()
    print(f'Execution time for DT is {end_time - start_time}')

    print('Starting iterations with SVM....')
    start_time = time.time()
    results_support_vector_machine = get_results(df, seeds, 'SVM')
    end_time = time.time()
    print(f'Execution time for SVM is {end_time - start_time}')

    results = {'DF': results_deep_forest,
               'DNN': results_neural_network,
               'RF': results_random_forest,
               'DT': results_decision_tree,
               'SVM': results_support_vector_machine}

    return results


def get_accuracy_dataframe(results):
    df_DF = results['DF'][1]
    df_DNN = results['DNN'][1]
    df_RF = results['RF'][1]
    df_DT = results['DT'][1]
    df_SVM = results['SVM'][1]

    df_results_DF = pd.DataFrame()
    df_results_DF['Accuracy'] = df_DF.mean(axis=1)
    df_results_DF['Model'] = 'DF'

    df_results_DNN = pd.DataFrame()
    df_results_DNN['Accuracy'] = df_DNN.mean(axis=1)
    df_results_DNN['Model'] = 'DNN'

    df_results_RF = pd.DataFrame()
    df_results_RF['Accuracy'] = df_RF.mean(axis=1)
    df_results_RF['Model'] = 'RF'

    df_results_DT = pd.DataFrame()
    df_results_DT['Accuracy'] = df_DT.mean(axis=1)
    df_results_DT['Model'] = 'DT'

    df_results_SVM = pd.DataFrame()
    df_results_SVM['Accuracy'] = df_SVM.mean(axis=1)
    df_results_SVM['Model'] = 'SVM'

    df_accuracy_models = pd.concat([df_results_DF, df_results_DNN, df_results_RF, df_results_DT, df_results_SVM])
    df_accuracy_models = df_accuracy_models.reset_index()
    df_accuracy_models = df_accuracy_models.rename(columns={'index': 'Split'})
    df_accuracy_models['text'] = round(df_accuracy_models['Accuracy'], 2).astype(str)

    return df_accuracy_models


def get_df_accuracy_models(results):
    df = pd.DataFrame()
    df['DF'] = results['DF'][1].mean()
    df['DNN'] = results['DNN'][1].mean()
    df['RF'] = results['RF'][1].mean()
    df['DT'] = results['DT'][1].mean()
    df['SVM'] = results['SVM'][1].mean()
    return df
