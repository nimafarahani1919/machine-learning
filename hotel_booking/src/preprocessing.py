
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder


def preprocess(numerical_columns:list,categorical_columns:list)->ColumnTransformer:
    numerical_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy = "median")),
        ("scaler",MinMaxScaler())
    ])
    categorical_pipeline = Pipeline(
        [
            ("imputer",SimpleImputer(strategy = "most_frequent")),
            ("encoder",OneHotEncoder(handle_unknown = "ignore"))
        ]
    )
    preprocessor = ColumnTransformer([
        ("num",numerical_pipeline,numerical_columns),
        ("cat",categorical_pipeline,categorical_columns)
    ])
    return preprocessor