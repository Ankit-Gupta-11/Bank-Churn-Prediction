from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 

import config
# import hyperparameter_tunning


def get_model(model):

    models = {"xgboost": XGBClassifier(),
                "logistic_regression": LogisticRegression(max_iter = 500),
                "random_forest": RandomForestClassifier(),
                "svm": SVC(probability=True)}

    
    if config.OPTIMIZATION == False:
        model = models[model]
        return model

    # elif config.OPTIMIZATION == True:
        
    #     best_params = hyperparameter_tunning.get_best_params(model)
    #     model = models[model](**best_params)

    #     return model 
    



