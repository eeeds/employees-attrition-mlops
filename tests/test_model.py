import pickle

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction import DictVectorizer


def test_extract_data():
    """
    Extract data from csv file and return dataframe

    Returns:
        pd.DataFrame: dataframe with data

    """
    df = pd.read_csv('./datasets/HR-Employee-Attrition.csv')
    # Delete unnecessary columns
    df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, inplace=True)
    # Changing categorical data to numerical data
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)

    assert df.shape == (1470, 32)


# def test_create_pipeline(train_dicts, y_train):
#     """
#     Create a pipeline to train a model.
#     Args:
#         train_dicts : list of dicts
#             The list of dictionaries to use for training.
#         y_train : list of floats
#             The list of target values to use for training.
#     Returns:
#         sklearn.pipeline.Pipeline:The pipeline to use for training.
#     """
#     pipeline = make_pipeline(
#         DictVectorizer(),
#         MaxAbsScaler(),
#         LogisticRegression(),
#     )
#     pipeline.fit(train_dicts, y_train)
#     # Save the pipeline to a file
#     with open("models/pipeline.bin", "wb") as f:
#         pickle.dump(pipeline, f)

#     assert 1==1
