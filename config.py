import os

INPUTS = "./inputs"
OUTPUTS = "./outputs"
FILE_NAME = "BankChurners"
TRAIN_DATA_PATH = os.path.join(INPUTS, f"{FILE_NAME}.csv")

FOLDS = 5
PROBLEM_TYPE = "classification"
TARGET_VARIABLE = "Attrition_Flag"

OPTIMIZATION = False  # if True, will use Bayesian optimization with gaussian process to find the best hyperparameters

MODEL_NAME = "xgboost"
MODEL_PATH = "./models"

THRESHOLD = 0.5