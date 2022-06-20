import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import os

import config


def handle_categorical_variables(df):
    
    # Making Dictionaries of ordinal features
    Income_Category_map = {
    'Less than $40K' : 0,
    '$40K - $60K'    : 1,
    '$60K - $80K'    : 2,
    '$80K - $120K'   : 3,
    '$120K +'        : 4,
    'Unknown'        : 5
    }


    Card_Category_map = {
        'Blue'     : 0,
        'Silver'   : 1,
        'Gold'     : 2,
        'Platinum' : 3
    }


    Attrition_Flag_map = {
        'Existing Customer' : 0,
        'Attrited Customer' : 1
    }

    Education_Level_map = {
        'Uneducated'    : 0,
        'High School'   : 1,
        'College'       : 2,
        'Graduate'      : 3,
        'Post-Graduate' : 4,
        'Doctorate'     : 5,
        'Unknown'       : 6
    }


    df.loc[:, 'Income_Category'] = df['Income_Category'].map(Income_Category_map)
    df.loc[:, 'Card_Category'] = df['Card_Category'].map(Card_Category_map)
    df.loc[:, 'Attrition_Flag'] = df['Attrition_Flag'].map(Attrition_Flag_map)
    df.loc[:, 'Education_Level'] = df['Education_Level'].map(Education_Level_map)


    #encoding city feature using label encoder
    lbe = LabelEncoder()

    cat_cols = [x for x in df.columns if df[x].dtype == 'object']

    for c in cat_cols:
        df.loc[:, c] = lbe.fit_transform(df.loc[:, c]) 

    df = df.sample(frac = 1).reset_index(drop = True) #To shuffle data

    df.drop(['CLIENTNUM'], axis = 1, inplace = True)
    
    return df



def preprocessing(df_pre):
    df_pre = handle_categorical_variables(df_pre)
    return df_pre


if __name__ == '__main__':
        
    # Loading the training data
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    df = preprocessing(df)

    df.to_csv(os.path.join(config.OUTPUTS, "BankChurners_processed.csv"), index = False)
