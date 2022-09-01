# Import libraries
import os
import sys
import pickle
from datetime import datetime

import mlflow
import pandas as pd
import whylogs as why
from prefect import flow, task
from whylogs.app import Session
from whylogs.proto import ModelType
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from evidently.dashboard import Dashboard
from whylogs.app.writers import WhyLabsWriter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from evidently.dashboard.tabs import ClassificationPerformanceTab
from sklearn.feature_extraction import DictVectorizer
from evidently.pipeline.column_mapping import ColumnMapping

from keys import keys_apis

mlflow.set_tracking_uri("sqlite:///mydb.sqlite")
EXPERIMENT_NAME = "hr-employee-attrition-project"
mlflow.set_experiment(EXPERIMENT_NAME)


@task(name='Performance Metrics', retries=3)
def performance_metrics(X_val, y_val, y_pred, session, logreg):
    """
    This function calculates the performance metrics and save it into Whylabs.
    Args:
        X_val (pandas.DataFrame): Validation features.
        y_val (pandas.DataFrame): Validation labels.
        y_pred (pandas.DataFrame): Predicted labels.
        session (whylogs.app.Session): Whylogs session.
        logreg (sklearn.linear_model.LogisticRegression): Logistic regression model.
    Returns:
        None
    """
    scores = [max(p) for p in logreg.predict_proba(X_val)]
    with session.logger(tags={"datasetId": "model-1"}, dataset_timestamp=datetime.now()) as ylog:
        ylog.log_metrics(
            targets=list(y_val),
            predictions=list(y_pred),
            scores=scores,
            model_type=ModelType.CLASSIFICATION,
            target_field="Attrition",
            prediction_field="prediction",
            score_field="Normalized Prediction Probability",
        )
    # closing the session
    session.close()


@task(name='Starting Whylogs', retries=3)
def starting_whylogs():
    """
    This function starts the Whylogs session.
    Args:
        None
    Returns:
        writer (whylogs.app.writers.WhyLabsWriter): Whylogs writer.
        session (whylogs.app.Session): Whylogs session.
    """
    k = keys_apis.Keys()
    k.obtain_whylogs_key()
    os.environ["WHYLABS_API_KEY"] = k.whylog_key
    os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-tgNtgy"
    # Adding the WhyLabs Writer to utilize WhyLabs platform
    writer = WhyLabsWriter("", formats=[])
    session = Session(project="model-1", pipeline="mlops-project-pipeline", writers=[writer])
    return writer, session


@task(name="Model Performance Dashboard", retries=3)
def model_performance_dashboard(df_train, train_dicts, df_val, val_dicts, numerical_features, categorical_features):
    """
    This function creates a dashboard that shows the performance of the model.
    Args:
        df_train (pandas.DataFrame): Training dataframe.
        train_dicts (list): List of dictionaries with the training data.
        df_val (pandas.DataFrame): Validation dataframe.
        val_dicts (list): List of dictionaries with the validation data.
        numerical_features (list): List of numerical features.
        categorical_features (list): List of categorical features.
    Returns:
        None, but it creates a dashboard and saves it in the folder called 'dashboards'.
    """
    df_column_mapping = ColumnMapping()
    df_column_mapping.target = "target"
    df_column_mapping.prediction = "prediction"
    df_column_mapping.numerical_features = numerical_features
    df_column_mapping.categorical_features = categorical_features
    # Prediction for dashboard
    with open("models/pipeline.bin", "rb") as f:
        pipeline = pickle.load(f)
    df_train["prediction"] = pipeline.predict(train_dicts)
    df_val["prediction"] = pipeline.predict(val_dicts)
    df_train.rename(columns={"Attrition": "target"}, inplace=True)
    df_val.rename(columns={"Attrition": "target"}, inplace=True)
    # Model Performance Dashboard full (verbose_level=1)
    df_model_performance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab(verbose_level=1)])
    df_model_performance_dashboard.calculate(df_train, df_val, column_mapping=df_column_mapping)
    # Save dashboard
    df_model_performance_dashboard.save("dashboards/df_model_performance.html")


@task(name="Create Pipeline", retries=3)
def create_pipeline(train_dicts, y_train):
    """
    Create a pipeline to train a model.
    Args:
        train_dicts : list of dicts
            The list of dictionaries to use for training.
        y_train : list of floats
            The list of target values to use for training.
    Returns:
        sklearn.pipeline.Pipeline:The pipeline to use for training.
    """
    pipeline = make_pipeline(
        DictVectorizer(),
        MaxAbsScaler(),
        LogisticRegression(),
    )
    pipeline.fit(train_dicts, y_train)
    # Save the pipeline to a file
    with open("models/pipeline.bin", "wb") as f:
        pickle.dump(pipeline, f)


@task(name="Extract_Data", retries=3)
def extract_data(writer, session) -> pd.DataFrame:
    """
    Extract data from csv file and return dataframe
    Returns:
        pd.DataFrame: dataframe with data
    """
    df = pd.read_csv("datasets/HR-Employee-Attrition.csv")
    # Delete unnecessary columns
    df.drop(["EmployeeCount", "EmployeeNumber", "StandardHours"], axis=1, inplace=True)
    # Changing categorical data to numerical data
    df["Attrition"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
    df["Over18"] = df["Over18"].apply(lambda x: 1 if x == "Yes" else 0)
    df["OverTime"] = df["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)

    with session.logger(tags={"datasetId": "model-1"}, dataset_timestamp=datetime.now()) as ylog:
        ylog.log_dataframe(df)
    return df


@task(name="Transform_data", retries=3)
def transform_data(df: pd.DataFrame):
    """
    Transform dataframe to get features and labels
    Args:
        df (pd.DataFrame): dataframe with data
    Returns:
        X_train (csr_matrix): features for training
        y_train (array): labels for training
        X_val (csr_matrix): features for validation
        y_val (array): labels for validation
    """
    # Categorical data
    categorical = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
    ]
    # Numerical data
    numerical = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "Education",
        "EnvironmentSatisfaction",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "Over18",
        "OverTime",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]
    ## Divide the data into train and test
    df_train_all, df_test = train_test_split(df, test_size=0.25, random_state=0)
    ##Obtain y values
    y_train_all = df_train_all["Attrition"].astype(int).values
    y_test = df_test["Attrition"].astype(int).values
    ## Training model
    df_train, df_val = train_test_split(df_train_all, test_size=0.25, random_state=0)
    y_train = df_train["Attrition"].astype(int).values
    y_val = df_val["Attrition"].astype(int).values
    ## Use DictVectorizer()
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    ## Applying MaxAbsScaler() to the data
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        df_train,
        df_val,
        numerical,
        categorical,
        train_dicts,
        val_dicts,
    )


@flow(name="Applying ML Model")
def applying_model():
    """
    Apply model to data
    Returns:
        None
    """
    writer, session = starting_whylogs()
    df = extract_data(writer, session)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        df_train,
        df_val,
        numerical,
        categorical,
        train_dicts,
        val_dicts,
    ) = transform_data(df)
    with mlflow.start_run():
        # Create tags and log params
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("developer", "Esteban")
        mlflow.log_param("train-data-path", "datasets/employee_data.csv")
        mlflow.log_param("val-data-path", "datasets/employee_data.csv")
        # Create Model
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(local_path="models/logreg.pkl", artifact_path="models/logreg")
        # Model Register
        mlflow.sklearn.log_model(
            sk_model=logreg,
            artifact_path="models/logreg",
            registered_model_name="sk-learn-logreg-model",
        )
    # Call create_pipeline()
    create_pipeline(train_dicts, y_train)
    # Create a model_performance_dashboard
    model_performance_dashboard(df_train, train_dicts, df_val, val_dicts, numerical, categorical)
    # Capture permorfance metrics to show
    performance_metrics(X_val, y_val, y_pred, session, logreg)

    return logreg


if __name__ == "__main__":
    """
    When you run this python script from the command line, it will run the flow
    Args:
        None
    Returns:
        None
    """
    logreg = applying_model()
    # Save model to pickle file
    with open("models/logreg.pkl", "wb") as f:
        pickle.dump(logreg, f)
    print("Model has been trained and saved")
