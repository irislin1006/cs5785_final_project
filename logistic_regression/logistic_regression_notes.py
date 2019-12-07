#!/usr/bin/env python3
from pprint import pprint
from time import time
import pickle
from in_hospital_mortality.custom_metrics import mortality_rate_at_k, train_val_compute
from in_hospital_mortality.feature_definitions import BOWFeatures, DictFeatures
from sklearn.preprocessing import MinMaxScaler #, StandardScalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--period_length', type=float, default=24.0, help='specify the period of prediction',
                        choices=[24.0, 48.0, -1])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='/data/joe/physician_notes/mimic-data/in_hospital_retro/')
    parser.add_argument('--feature_period', type=str, help='feature period',
                        choices=["24", "48", "retro"])
    parser.add_argument('--feature_used', type=str, help='feature used',
                        choices=["all", "notes", "all_but_note"])
    parser.add_argument('--balanced', dest="balanced", action="store_true", help = 'whether to use balanced class weights')
    args = parser.parse_args()
    args.period_length = float('inf') if args.period_length == -1 else args.period_length
    print (args)

    datapath = args.data
    featurepath = f'/data/joe/physician_notes/mimic-data/preprocessed/features_{args.feature_period}.pkl'
    train_list = pd.read_csv(os.path.join(datapath, "train", "listfile.csv"))
    test_list = pd.read_csv(os.path.join(datapath, "test", "listfile.csv"))

    print("Loading data")
    X_train_notes, y_train, X_test_notes, y_test = [], [], [], []
    t=time()
    features = pickle.load(open(featurepath,'rb'))
    for index, row in train_list.iterrows():
        note = pd.read_csv(os.path.join(datapath,'train', row['stay']))
        note = " ".join(note['900002'].values)
        X_train_notes.append(note)
        y_train.append(row['y_true'])

    for index, row in test_list.iterrows():
        note = pd.read_csv(os.path.join(datapath,'test', row['stay']))
        note = " ".join(note['900002'].values)
        X_test_notes.append(note)
        y_test.append(row['y_true'])

    X_train_notes, X_val_notes, y_train, y_val, X_train_names, X_val_names = \
            train_test_split(X_train_notes, y_train, train_list['stay'].values, test_size=0.2, random_state=19)

    train_notes = pd.DataFrame({'file_name': X_train_names, 'text': X_train_notes})
    val_notes = pd.DataFrame({'file_name': X_val_names, 'text': X_val_notes})
    test_notes = pd.DataFrame({'file_name': test_list['stay'].values, 'text': X_test_notes})
    union_list = []
    if args.feature_used in ['all', 'notes']:
        print ("add Bag of Words features .....")
        union_list.append(("tfidf", BOWFeatures()))
    if args.feature_used in ['all','all_but_notes']:
        print ("add structured variable features ..... ")
        union_list.append(("structured",
                           Pipeline([
                               ("fe", DictFeatures(features)),
                               ("imputer", SimpleImputer()),
                               ("scaler", MinMaxScaler()),
                           ])))

    print("Total number of training data:", len(X_train_notes))
    print("Total number of validation data:", len(X_val_notes))
    print("Total number of test data:", len(X_test_notes))
    print("data loading time:", time()-t)

    pipeline = Pipeline([
        ('union', FeatureUnion(union_list)),
        ('lr', LogisticRegression(solver="liblinear", max_iter = 500,
                                  class_weight="balanced" if args.balanced else None)),
    ])

    parameters = {
        "lr__C": np.logspace(-8, 3, 11, base = 2)
    }

    # Display of parameters

    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    # Training on training data and hyperparameter tuning on validation data

    t0 = time()
    pipeline, best_score, best_parameters, params, scores = train_val_compute(train_notes, val_notes, y_train, y_val, pipeline, parameters)
    print("done in %0.3fs" % (time() - t0))
    print()

    # Displaying training results

    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print ("Mean test score:")
    print(scores)
    print("Best score: \n%0.3f" % best_score)

    """
    # Displaying test results

    val_predicted = pipeline.predict_proba(val_notes)[:, 1]
    print ("ROC AUC Score on Test Set:")
    print(roc_auc_score(y_val, val_predicted))

    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(y_val, val_predicted, K))
    """
