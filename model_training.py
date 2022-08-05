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
from scipy import stats


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


def train_deep_forest(x_train, y_train, params):
    """
    Returns a trained Deep Forest with the given data.
    There is a bug in the code for Deep Forest and the
    classifier can't be reused in a loop.
    """
    clf = CascadeForestClassifier(**params)

    clf.fit(x_train, y_train)
    return clf


def train_neural_network(x_train, y_train, params):
    """
    Returns a trained Neural Network with the given data.
    There is a distinction between binary or multiclass case
    but the Neural Network architecture is the same.
    """
    nodes = params['nodes']
    dropout = params['dropout']
    learning_rate = params['learning_rate']
    activation = params['activation']
    epochs = params['epochs']

    n_classes = len(np.unique(y_train))
    n_features = x_train.shape[1]
    sequential_list = [InputLayer(n_features),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal()),
                       Dropout(dropout),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal()),
                       Dropout(dropout),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal()),
                       Dropout(dropout),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal()),
                       Dropout(dropout),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal()),
                       Dropout(dropout),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal()),
                       Dropout(dropout),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal()),
                       Dropout(dropout),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal()),
                       Dropout(dropout),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal()),
                       Dropout(dropout),
                       Dense(nodes, activation=activation,
                             kernel_initializer=tf.keras.initializers.HeNormal())
                       ]
    if n_classes > 2:
        sequential_list.append(Dense(n_classes + 1, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
    else:
        sequential_list.append(Dense(1, activation="sigmoid"))
        loss = 'binary_crossentropy'

    clf = Sequential(sequential_list)
    clf.compile(loss=loss,
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy'])
    clf.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)

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
        loss, accuracy = clf.evaluate(x, y, verbose = 0)

    else:
        hits = sum(prediction == y)
        total = len(y)
        accuracy = hits / total

    accuracy = 100 * round(accuracy, 2)

    return {'accuracy': accuracy, 'time': time_prediction}


def get_results(df, seeds, model, model_params):
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

    results = {'accuracy_train': pd.DataFrame(columns=cols),
               'accuracy_test': pd.DataFrame(columns=cols),
               'time_training': pd.DataFrame(columns=cols),
               'time_prediction_train': pd.DataFrame(columns=cols),
               'time_prediction_test': pd.DataFrame(columns=cols)}

    for seed in seeds:
        if (seed + 1) % 1 == 0:
            print(f'\n Calculating seed {seed + 1} out of {len(seeds)}')
            print('-------------------------')

        df_sample = get_sample(df, size, seed)
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
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            params = model_params[model]

            start_time = time.time()

            if model == 'DF':
                clf = train_deep_forest(x_train, y_train, params)

            elif model == 'DNN':
                clf = train_neural_network(x_train, y_train, params)

            elif model == 'RF':
                clf = RandomForestClassifier(**params)
                clf.fit(x_train, y_train)

            elif model == 'DT':
                clf = DecisionTreeClassifier(**params)
                clf.fit(x_train, y_train)

            elif model == 'SVM':
                clf = svm.SVC(**params)
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


def compute_all_models_results(df, model_params, n_seeds=30):
    """
    Computes the results for all the models
    """
    seeds = list(range(n_seeds))

    print('Starting iterations with DNN....')
    start_time = time.time()
    results_neural_network = get_results(df, seeds, 'DNN', model_params)
    end_time = time.time()
    print(f'\n Execution time for DNN is {end_time - start_time} \n')
    print('--------------------------------------')

    print('Starting iterations with DF....')
    start_time = time.time()
    results_deep_forest = get_results(df, seeds, 'DF', model_params)
    end_time = time.time()
    print(f'\n Execution time for DF is {end_time - start_time} \n')
    print('--------------------------------------')

    print('Starting iterations with RF....')
    start_time = time.time()
    results_random_forest = get_results(df, seeds, 'RF', model_params)
    end_time = time.time()
    print(f'\n Execution time for RF is {end_time - start_time} \n')
    print('--------------------------------------')

    print('Starting iterations with DT....')
    start_time = time.time()
    results_decision_tree = get_results(df, seeds, 'DT', model_params)
    end_time = time.time()
    print(f'Execution time for DT is {end_time - start_time} \n')

    print('Starting iterations with SVM....')
    start_time = time.time()
    results_support_vector_machine = get_results(df, seeds, 'SVM', model_params)
    end_time = time.time()
    print(f'Execution time for SVM is {end_time - start_time} \n')

    results = {'DF': results_deep_forest,
               'DNN': results_neural_network,
               'RF': results_random_forest,
               'DT': results_decision_tree,
               'SVM': results_support_vector_machine}

    return results


def get_average_results(results, metric):
    """
    Computes average for all iterations for the given metric.
    Returns a dataframe with the results
    """
    df = pd.DataFrame()
    df['DF'] = round(results['DF'][metric].mean(),4)
    df['DNN'] = round(results['DNN'][metric].mean(),4)
    df['RF'] = round(results['RF'][metric].mean(),4)
    df['DT'] = round(results['DT'][metric].mean(),4)
    df['SVM'] = round(results['SVM'][metric].mean(),4)
    df.index = df.index*100 + 100
    return df


def students_t_test(sample_1, sample_2):
    """ Computes pairwise t-student p-value for the given samples """
    return stats.ttest_ind(a=sample_1, b=sample_2, equal_var=True).pvalue


def compute_all_p_values(df):
    """
    Computes all sample t-student p-value combinations
    for all models
    """
    combinations_pairs = [('DF', 'DNN'), ('DF', 'RF'), ('DF', 'DT'),
                          ('DNN', 'RF'), ('DNN', 'DT'), ('RF', 'DT'),
                          ('DF', 'SVM'), ('DNN', 'SVM'), ('RF', 'SVM'),
                          ('DT', 'SVM')]

    for pair in combinations_pairs:
        print(pair)
        col_1 = pair[0]
        col_2 = pair[1]
        sample_1 = df[col_1]
        sample_2 = df[col_2]
        students_t_test(sample_1, sample_2)
        print('----------------------------')


def create_table_accuracy(results, metric):
    """
    Prints a table with accuracies and statistically
    significant means
    """
    model_means = {'DF': results['DF'][metric].mean(),
                   'DNN': results['DNN'][metric].mean(),
                   'RF': results['RF'][metric].mean(),
                   'DT': results['DT'][metric].mean(),
                   'SVM': results['SVM'][metric].mean()}

    df_table = pd.DataFrame(columns=model_means.keys())

    for n in range(0, 10):
        new_row = {model: [round(model_means[model][n].mean(), 2)]
                   for model in model_means}
        df_new_row = pd.DataFrame.from_dict(new_row)
        df_table = pd.concat([df_table, df_new_row], ignore_index=True)
        best_model = df_table.iloc[n].astype(float).idxmax()
        best_models = [best_model]

        for model in model_means:
            if model != best_model:
                p_value = students_t_test(results[best_model]['accuracy_test'][n],
                                          results[model]['accuracy_test'][n])
                if p_value > 0.05:
                    best_models.append(model)

        df_table = df_table.astype(str)

        for model in best_models:
            df_table.iloc[n][model] = df_table.iloc[n][model] + '*'

        df_table.iloc[n][best_model] = '\textcolor{blue}{\textbf{' + df_table.iloc[n][best_model] + '}}'

    df_table.index = df_table.index * 100 + 100

    return df_table
