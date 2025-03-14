
"""
==================================================
Machine Learning Model Training and Optimization
==================================================
This script provides functions for training, evaluating, and optimizing
various machine learning models (Random Forest, LGBM, SVM, XGBoost, MLP).
It includes:
- `apply_model`: A function to train and evaluate a specified model.
- `grid_search_cv`: A function to perform hyperparameter tuning using GridSearchCV.

Supported models:
- Random Forest (RF)
- LightGBM (LGBM)
- Support Vector Machine (SVM)
- XGBoost (XGBOOST)
- Multi-Layer Perceptron (MLP)

Author: JUILLARD Thibaut and GHIZLAN Moqim


@@ ChatGPT and Github copilot were used to write this code specifically to generate the documentation and the comments.
@@ The function grid_search_cv might not work as expected, due to the modifications applied after. The results might not be the same as expected.
"""

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV


def apply_model(model_name, X_train, y_train, X_test, y_test, **kwargs):
    """
    Train and evaluate a machine learning model.

    Parameters:
    - model_name (str): The name of the model to train. Options: 'RF', 'LGBM', 'SVM', 'XGBOOST', 'MLP'.
    - X_train (array-like): Training feature set.
    - y_train (array-like): Training labels.
    - X_test (array-like): Testing feature set.
    - y_test (array-like): Testing labels.
    - **kwargs: Additional hyperparameters for the model.

    Returns:
    - tuple:
        - classification_report (str): Performance report including precision, recall, and F1-score.
        - f1_score (float): Weighted F1-score of the model.
        - y_pred (array-like): Predicted labels for the test set.
    """

    models = {
        'RF': RandomForestClassifier,
        'LGBM': LGBMClassifier,
        'SVM': SVC,
        'XGBOOST': xgb.XGBClassifier,
        'MLP': MLPClassifier
    }

    # Ensure the model name is valid
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")

    # Special handling for XGBoost
    if model_name == "XGBOOST":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Remove `num_boost_round` from params dictionary before passing it to `train`
        params = {key: value for key, value in kwargs.items() if key != 'num_boost_round'}

        model = xgb.train(params, dtrain, num_boost_round=kwargs.get('num_boost_round', 100))

        # Ensure predictions are discrete class labels (Fix for classification error)
        y_pred = model.predict(dtest)
        y_pred = np.round(y_pred).astype(int)  # Convert continuous outputs to integer class labels
    else:
        # Instantiate model and train
        model = models[model_name](**kwargs)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

    # Compute classification report and F1-score
    return (
        classification_report(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, average="weighted"),
        y_pred
    )


def grid_search_cv(model_name, X_train, y_train, X_test, y_test, params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=42):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
    - model_name (str): Name of the model ('RF', 'LGBM', 'SVM', 'XGBOOST', 'MLP').
    - X_train (array-like): Training feature set.
    - y_train (array-like): Training labels.
    - X_test (array-like): Testing feature set.
    - y_test (array-like): Testing labels.
    - params (dict): Dictionary of hyperparameters to tune.
    - cv (int, default=3): Number of cross-validation folds.
    - scoring (str, default='f1_weighted'): Scoring metric for evaluation.
    - n_jobs (int, default=-1): Number of parallel jobs for training.
    - verbose (int, default=1): Level of verbosity for GridSearchCV.
    - random_state (int, default=42): Random seed for reproducibility.

    Returns:
    - tuple:
        - classification_report (str): Performance report for the best model.
        - best_params (dict): Optimal hyperparameters found.
        - f1_score (float): Weighted F1-score of the best model.
    """

    # Define available models
    def get_model_by_name(model_name):
        models = {
            'RF': RandomForestClassifier(random_state=random_state),
            'LGBM': lgb.LGBMClassifier(random_state=random_state),
            'SVM': SVC(random_state=random_state),
            'XGBOOST': xgb.XGBClassifier(random_state=random_state),
            'MLP': MLPClassifier(random_state=random_state),
        }
        return models.get(model_name, None)

    # Retrieve the model class
    model = get_model_by_name(model_name)
    if model is None:
        raise ValueError(f"Unsupported model: {model_name}")

    # Perform GridSearchCV
    grid = GridSearchCV(model, params, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    grid.fit(X_train, y_train)

    # Retrieve the best model
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    best_params = grid.best_params_

    # Compute classification report and F1-score
    return (
        classification_report(y_test, y_pred, zero_division=0),
        best_params,
        f1_score(y_test, y_pred, average="weighted")
    )
