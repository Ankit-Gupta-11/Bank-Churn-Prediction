import pandas as pd 
import numpy as np  
import os
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
import joblib 
import argparse


import config
import model_dispatcher


def run(fold, model_list):
    df = pd.read_csv(os.path.join(config.OUTPUTS, f"{config.FILE_NAME}_kfold.csv"))
    
    df_train = df[df['kfold'] != fold].sample(frac = 1).reset_index(drop = True)
    df_val = df[df['kfold'] == fold].sample(frac = 1).reset_index(drop = True)


    X_train = df_train.drop(['kfold', config.TARGET_VARIABLE], axis = 1)
    y_train = df_train[config.TARGET_VARIABLE]

    X_val = df_val.drop(['kfold', config.TARGET_VARIABLE], axis = 1)
    y_val = df_val[config.TARGET_VARIABLE]
    
    clfs = []
    for model in model_list:
        clfs.append(model_dispatcher.get_model(model))

    estimators = []
    for name, clf in zip(model_list, clfs):
        estimators.append((name, clf))

    print(f"Applying Voting Classifier using {model_list} on fold {fold}")
    model = VotingClassifier(
            #    estimators = [('lr', clf1), ('svm', clf2), ('rf', clf3), ('xgb', clf4)],
                estimators = estimators,
               voting = 'soft')
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, preds)

    model_names = " ,".join(model_list)
    print(f"FOLD {fold}, Voting Classifier of {model_names}, AUC: {score}")

    # save the model
    joblib.dump(model, 
                os.path.join(config.MODEL_PATH, f"VotingClassifier_{fold}.bin"))

    return score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type = int)

    parser.add_argument('--model', help='delimited list input', type=str)

    args = parser.parse_args()

    model_list = [item for item in args.model.split(',')]
    run(args.fold, model_list)