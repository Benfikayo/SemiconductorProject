import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, StratifiedKFold

from src.SemiconductorProject.exceptions import CustomException
from src.SemiconductorProject.logger import logging


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The Python object to be saved.

    Raises:
        CustomException: If there is an error while saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
            logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load a Python object from a file using pickle.
Args:
        file_path (str): The path to the file from which the object will be loaded.

    Returns:
        object: The Python object loaded from the file. 

    Raises:
        CustomException: If there is an error while loading the object.
    """
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error occurred while loading object: {e}")
        raise CustomException(e, sys)
def group_train_test_split(X, y, groups, test_size=0.2, random_state=None):
    """
    Perform a group-wise train-test split.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        groups (pd.Series): Group labels for the samples.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Split data as (X_train, X_test, y_train, y_test).
    """
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        logging.info("Group-wise train-test split completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error occurred during group-wise train-test split: {e}")
        raise CustomException(e, sys)


def evaluate_model(X_train, X_test, y_train, y_test, model: dict, param_grids: dict=None):

    try:
        reports = {}
        for name, model in model.items():
            logging.info(f"Evaluating model: {name}")
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test) 

            reports[name] = {

                "model": model,
                "train_r2_score": r2_score(y_train, y_train_pred),
                "test_r2_score": r2_score(y_test, y_test_pred),
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "y_train_pred": y_train_pred,
                "y_test_pred": y_test_pred,
            }
            logging.info(f"{name}: Test R2 Score: {reports[name]['test_r2_score']:.4f}, Test MAE: {reports[name]['test_mae']:.4f},  Train R2 Score: {reports[name]['train_r2_score']:.4f}, Train MAE: {reports[name]['train_mae']:.4f}"
                         )
        return reports
    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {e}")
        raise CustomException(e, sys)

