# Libraries
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Metrics
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, root_mean_squared_error,
                             f1_score, precision_score, recall_score, roc_auc_score, mean_absolute_error)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

# Fine-Tuning / Model-Selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split

# Regression Models
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
                              VotingRegressor, BaggingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Classification Models
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier,
                              VotingClassifier, GradientBoostingClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import ShuffleSplit

# Clustering Models
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch, OPTICS


# Function used to detect Outliers , their counts and percentages
def analyze_IQR_outliers(data, num_columns):
    """
    Analyzes outliers in numerical columns of a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        num_columns (list): A list of numerical column names to analyze.

    Returns:
        dict: A dictionary containing outliers, their counts and percentages for each column.
    """

    outlier_data = {}
    for col in num_columns:
        data_column = data[col]
        Q1 = data_column.quantile(0.25)
        Q3 = data_column.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data_column[(data_column < lower_bound) | (data_column > upper_bound)]
        outlier_counts = len(outliers)
        outlier_percentages = (outlier_counts / len(data_column)) * 100

        outlier_data[col] = {
            'count': outlier_counts,
            'percentage': outlier_percentages,
            'outliers': outliers.tolist()
        }

        if outlier_counts > 0:
            print(f"No. of IQR outliers in '{col}': {outlier_counts}")
            print(f"Percentage of outliers in '{col}': {outlier_percentages:.2f}%")
            print("-" * 80)
    return outlier_data


# Function to visualize counts, percentages outliers or both
def visualize_outliers(outlier_data, plot_type="count"):
    """
    Visualizes outliers in numerical columns of a DataFrame using Plotly.

    Args:
        outlier_data (dictionary): The dictionary containing the outliers, their counts, and percentages
                                   for each column.
        plot_type (str, optional): Controls the type of plot to show. Valid options are
                                   "counts", "percentages". Defaults to "count".

    Returns:
        Plotly visualizations of outliers in numerical columns.
    """

    # Extract the data for plotting
    columns = list(outlier_data.keys())
    counts = sorted([value['count'] for value in outlier_data.values()], reverse=True)
    percentages = sorted([round(value['percentage'], 2) for value in outlier_data.values()], reverse=True)

    # Visualization based on plot_type argument
    if plot_type == "counts":
        fig = px.bar(x=columns, y=counts, color=columns, text_auto=True)
        fig.update_layout(
            title='Number of Outliers in Each Column',
            xaxis_title='Columns',
            yaxis_title='Count',
            xaxis_tickangle=-90,
            height=800
        )
        fig.show()

    elif plot_type == "percentages":
        fig = px.bar(x=columns, y=percentages, color=columns, text_auto=True)
        fig.update_layout(
            title='Percentage of Outliers in Each Column',
            xaxis_title='Columns',
            yaxis_title='Percentage',
            xaxis_tickangle=-90,
            height=800
        )
        fig.show()

    else:
        print(f"Invalid plot_type: {plot_type}. Valid options are 'counts', 'percentages', or 'both'.")


# Replace outliers in a column  with specific value (handle missing values appropriately)
def replace_outliers(data, column, value_to_replace):
    """
    Imputes outliers in each column of a DataFrame with a specific value.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (list): A dictionary containing outlier information for each column,
                             returned by the analyze_visualize_outliers function.
        value_to_replace(int): The value to replace the outlier.
    Returns:
        pd.DataFrame: A new DataFrame with outliers imputed using the value_to_replace.
    """

    imputed_data = data.copy()  # Create a copy to avoid modifying original data

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = imputed_data[column].quantile(0.25)
    Q3 = imputed_data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define the bounds for non-outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with the boundary values
    imputed_data[column] = np.where(imputed_data[column] < lower_bound, value_to_replace, imputed_data[column])
    imputed_data[column] = np.where(imputed_data[column] > upper_bound, value_to_replace, imputed_data[column])

    return imputed_data


# Calculate the percentage of null values in each colum
def visualize_nulls(data, plot_type="count"):
    """
        Visualizes Missing values in the DataFrame using Plotly.

        Args:
            data (pd.Dataframe): The dataframe containing all the data
            plot_type (str, optional): Controls the type of plot to show. Valid options are
                                       "counts", "percentages". Defaults to "count".

        Returns:
            Plotly visualizations of outliers in numerical columns.
    """

    nulls = data.isna().sum(axis=0)

    # Filter out columns with no missing values
    nulls = nulls[nulls > 0]

    nulls_df = nulls.reset_index()

    nulls_df.columns = ['Column', "Null Count"]
    nulls_df["Null Percentage"] = round(nulls_df["Null Count"] / len(data), 4) * 100

    if plot_type == "count":
        fig = px.bar(nulls_df.sort_values(by="Null Count", ascending=True),
                     y='Column',
                     X='Null Count',
                     title='Count of Missing Values in Each Column',
                     color_discrete_sequence=['red'],
                     text_auto=True)

        fig.update_layout(yaxis_title='Columns',
                          xaxis_title='Missing Values Count',
                          show_legend=False,
                          margin=dict(t=50, l=100),
                          height=400 + len(nulls_df) * 10)

        fig.show()
    elif plot_type == "percentage":
        fig = px.bar(nulls_df.sort_values(by="Null Percentage", ascending=True),
                                      y='Column',
                                      X='Null Percentage',
                                      title='Percentage of Missing Values in Each Column',
                                      color_discrete_sequence=['red'],
                                      text_auto=True)

        fig.update_layout(yaxis_title='Columns',
                      xaxis_title='Missing Values Percentage',
                      show_legend=False,
                      margin=dict(t=50, l=100),
                      height=400 + len(nulls_df) * 10)

        fig.show()

    return nulls_df

    # Function used to evaluate a regression model


def evaluate_regression_models(X_train, y_train, X_test, y_test, models):
    """
  Evaluates a list of models, stores results, and returns a DataFrame for comparison and trained models.

  Args:
      X_train (pd.DataFrame): Training features.
      y_train (pd.Series): Training target variable.
      X_test (pd.DataFrame): Testing features.
      y_test (pd.Series): Testing target variable.
      models (list): A list of scikit-learn model objects.

  Returns:
      tuple: A tuple containing two elements:
          - pd.DataFrame: A DataFrame containing model names and evaluation metrics.
          - dict: A dictionary where keys are model names and values are the trained models.
    """

    model_results = []
    # trained_models = {}
    for model in models:
        model.fit(X_train, y_train)
        # trained_models[model.__class__.__name__] = model  # Save trained model with name
        start_time = time.time()  # Record start time
        prediction = model.predict(X_test)
        inference_time = time.time() - start_time  # Calculate inference time

        model_results.append({
            "Model-Name": model.__class__.__name__,
            # "MSE": mean_squared_error(y_test, prediction),
            "RMSE": root_mean_squared_error(y_test, prediction),
            "R2_Score": r2_score(y_test, prediction) * 100,
            "MAE": mean_absolute_error(y_test, prediction),
            "Inference Time (ms)": inference_time * 1000
        })

    models_df = pd.DataFrame(model_results)
    models_df = models_df.set_index('Model-Name')
    return models_df.sort_values("R2_Score", ascending=False)


# Function used to evaluate a classification model
def evaluate_classification_models(X_train, y_train, X_test, y_test, models):
    """
  Evaluates a list of models, stores results, and returns a DataFrame for comparison and trained models.

  Args:
      X_train (pd.DataFrame): Training features.
      y_train (pd.Series): Training target variable.
      X_test (pd.DataFrame): Testing features.
      y_test (pd.Series): Testing target variable.
      models (list): A list of scikit-learn model objects.

  Returns:
      tuple: A tuple containing two elements:
          - pd.DataFrame: A DataFrame containing model names and evaluation metrics.
          - dict: A dictionary where keys are model names and values are the trained models.
    """

    model_results = []
    # trained_models = {}
    for model in models:
        model.fit(X_train, y_train)
        # trained_models[model.__class__.__name__] = model  # Save trained model with name
        start_time = time.time()  # Record start time
        prediction = model.predict(X_test)
        inference_time = time.time() - start_time  # Calculate inference time

        model_results.append({
            "Model-Name": model.__class__.__name__,
            "Accuracy": accuracy_score(y_test, prediction) * 100,
            "ROC_AUC": roc_auc_score(y_test, prediction),
            "F1_Score": f1_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction),
            "Recall": recall_score(y_test, prediction),
            "Inference Time (ms)": inference_time * 1000
        })

    models_df = pd.DataFrame(model_results)
    models_df = models_df.set_index('Model-Name')
    return models_df.sort_values("F1_Score", ascending=False)


# Function used to plot confusion matrix and classification Report
def evaluate_classification_metrics(y_true, y_pred, target_names=None, display=True):
    """
    Evaluates a classification model by generating a confusion matrix, confusion matrix display (optional), and classification report.

    Args:
        y_true (pd.Series): Ground truth labels.
        y_pred (pd.Series): Predicted labels.
        target_names (list, optional): List of class names for improved readability of the confusion matrix. Defaults to None.
        display (bool, optional): Whether to display the confusion matrix visually using ConfusionMatrixDisplay. Defaults to True.

    Returns:
        dict: A dictionary containing the confusion matrix, classification report, and class names (if provided).
    """

    # Ensure y_true and y_pred are NumPy arrays for compatibility with sklearn metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=target_names)

    # Optionally display confusion matrix visually
    if display:
        ConfusionMatrixDisplay(cm, display_labels=target_names).plot()
        plt.show()  # Display the confusion matrix plot

    # Return results in a dictionary for easy access
    evaluation_results = {
        "Confusion Matrix": cm,
        "Classification Report": report,
        "Target Names": target_names,  # Include target names if provided
    }
    return evaluation_results


# Function used to evaluate a clustering model
def evaluate_clustering_models(X_train, X_test, models):
    """
    Evaluates a list of clustering models, stores results, and returns a DataFrame for comparison.

    Args:
        X_train (pd.DataFrame): Training features (used for fitting the model).
        X_test (pd.DataFrame): Testing features (used for silhouette score calculation).
        models (list): A list of scikit-learn clustering model objects.

    Returns:
        pd.DataFrame: A DataFrame containing model names and evaluation metrics (silhouette score).
      """

    model_results = []
    for model in models:
        model.fit(X_train)
        prediction = model.predict(X_test)
        silhouette = silhouette_score(X_test, prediction)  # Calculate silhouette score, Intra-cluster vs inter-cluster distance
        calinski_harabasz = calinski_harabasz_score(X_test, prediction)  # Between-cluster vs within-cluster variance
        davies_bouldin = davies_bouldin_score(X_test, prediction)  # Ratio of within-cluster scatter to separation of clusters

        model_results.append({
            "Model-Name": model.__class__.__name__,
            "Silhouette Score": silhouette,
            "Calinski-Harabasz Score": calinski_harabasz,
            "Davies-Bouldin Score": davies_bouldin,
        })

    models_df = pd.DataFrame(model_results)
    models_df = models_df.set_index('Model-Name')
    return models_df.sort_values("Silhouette Score", ascending=False)


def evaluate_models(X_train, y_train, X_test, y_test, models, task_type="regression"):
    """
  Evaluates a list of models, stores results, and returns a DataFrame for comparison.

  Args:
      X_train (pd.DataFrame): Training features.
      y_train (pd.Series): Training target variable.
      X_test (pd.DataFrame): Testing features.
      y_test (pd.Series): Testing target variable.
      models (list): A list of scikit-learn model objects.
      task_type (str, optional): "classification" or "regression" (default: "regression").

  Returns:
      pd.DataFrame: A DataFrame containing model names and evaluation metrics.
    """

    supported_tasks = ("classification", "regression", "clustering")
    if task_type not in supported_tasks:
        raise ValueError(f"Invalid task type: {task_type}. Supported types: {', '.join(supported_tasks)}")

    for model in models:
        model.fit(X_train, y_train)
        start_time = time.time()  # Record start time
        prediction = model.predict(X_test)
        inference_time = time.time() - start_time  # Calculate inference time

    model_results = []
    metric_functions = {
        "classification": {
            "Accuracy": accuracy_score,
            "F1-Score": f1_score,
            "Precision": precision_score,
            "Recall": recall_score,
            "AUC-ROC": roc_auc_score
        },
        "regression": {
            "RMSE": mean_squared_error(y_test, prediction, squared=False),
            "R2-Score": r2_score,
            "MAE": mean_absolute_error,
        },
        "clustering": {
            "Silhouette Score": silhouette_score,
            "Calinski-Harabasz Score": calinski_harabasz_score,
            "Davies-Bouldin Score": davies_bouldin_score,
        }
    }

    metrics = metric_functions[task_type]

    model_results.append({
        "Model-Name": model.__class__.__name__,
        **{metric: func(y_test, prediction) for metric, func in metrics.items()},
        "Inference Time (ms)": inference_time * 1000
    })

    models_df = pd.DataFrame(model_results)
    models_df = models_df.set_index('Model-Name')
    return models_df


# Example usage for classification:
classification_models = [
    LogisticRegression(random_state=42),
    DecisionTreeClassifier(random_state=42),
    ExtraTreeClassifier(random_state=42),
    XGBClassifier(random_state=42),
    XGBRFClassifier(random_state=42),
    AdaBoostClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    ExtraTreesClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    BaggingClassifier(random_state=42),
    SGDClassifier(random_state=42),
    SVC(random_state=42),
    KNeighborsClassifier(n_neighbors=3)
]

# Example usage for regression:
regression_models = [
    DecisionTreeRegressor(random_state=42),
    XGBRegressor(random_state=42),
    ExtraTreeRegressor(random_state=42),
    XGBRFRegressor(random_state=42),
    LinearRegression(),
    RandomForestRegressor(random_state=42),
    GradientBoostingRegressor(random_state=42),
    AdaBoostRegressor(random_state=42),
    BaggingRegressor(random_state=42),
    ExtraTreesRegressor(random_state=42),
    SGDRegressor(random_state=42),
    SVR(),
    KNeighborsRegressor(n_neighbors=3),
    Ridge(random_state=42),
    Lasso(random_state=42),
]

clustering_models = [
    # Centroid-based
    KMeans(n_clusters=5, random_state=42),  # Needs predefined number of clusters (k)

    # Hierarchical
    AgglomerativeClustering(n_clusters=5, linkage='ward'),  # Needs predefined number of clusters
    Birch(n_clusters=5),  # Can automatically determine number of clusters

    # Density-based
    DBSCAN(eps=0.5, min_samples=5),  # Adjust eps and min_samples based on your data
    OPTICS(min_samples=5, eps=0.5),  # Similar to DBSCAN but finds clusters of varying density

    # Partitioning around medoids (PAM) - useful for noisy data
    # PAM(n_clusters=5),

    # Model-based (uses statistical models to represent clusters)
    GaussianMixture(n_components=5),  # Adjust n_components for number of clusters

    # Spectral clustering (uses spectral properties of a similarity graph)
    SpectralClustering(n_clusters=5),  # Needs number of clusters
]


# Function used to implement voting classifier with top 3 classification models
def get_voting(models_df, n_top=3, voting_type='classifier'):
    """
   Creates a voting ensemble (classifier or regressor) using the top n_top models
   from a DataFrame containing model prediction scores.

   Args:
       models_df (pandas.DataFrame): A DataFrame with columns for model names
           and prediction scores, sorted with the highest scores at the top.
       n_top (int, optional): The number of top models to use in the ensemble.
           Defaults to 3.
       voting_type (str, optional): 'classifier' or 'regressor' to specify the
           type of ensemble to create. Defaults to 'classifier'.

   Returns:
       VotingClassifier or VotingRegressor: The appropriate ensemble model.

   Raises:
       ValueError: If voting_type is not 'classifier' or 'regressor'.
   """

    # Extract model names and prediction columns
    top_models = models_df.head(n_top)['Model-Name'].tolist()  # Assuming 'Model-Name' column for both cases
    prediction_columns = [col for col in models_df.columns if col != 'Model-Name']

    # Create a dictionary of estimators
    estimators = {model: models_df[col] for model, col in zip(top_models, prediction_columns)}

    # Create the appropriate voting ensemble based on voting_type
    if voting_type == 'classifier':
        ensemble = VotingClassifier(estimators=estimators, voting='hard')
    elif voting_type == 'regressor':
        ensemble = VotingRegressor(estimators=estimators)
    else:
        raise ValueError(f"Invalid voting_type: {voting_type}. Must be 'classifier' or 'regressor'.")

    return ensemble


def accuracy_and_rmse(y_test, prediction):
    print('accuracy: ' + str(accuracy_score(prediction, y_test) * 100) + " %")
    lin_rmse = root_mean_squared_error(y_test, prediction)
    print('\nrmse: ' + str(lin_rmse))  # rmse


def precision_recall_f1(y_test, prediction):
    print('precision: ' + str(precision_score(y_test, prediction)))
    print('recall:    ' + str(recall_score(y_test, prediction)))
    print('F1_score:  ' + str(f1_score(y_test, prediction)))


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label='auc= ' + str(label))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate(Recall)")
    plt.title("ROC Curve")
    plt.axis([0, 1, 0, 1])
    plt.legend(loc=4)
    plt.show()


def grid_search_classification_models(X, y):
    models = [
        ('LogisticRegression', LogisticRegression(random_state=42)),
        ('RandomForest', RandomForestClassifier(random_state=42)),
        ('DecisionTree', DecisionTreeClassifier(random_state=42)),
        ('ExtraTrees', ExtraTreesClassifier(random_state=42)),
        ('XGBoost', XGBClassifier()),
        ('XGBoostRandomForest', XGBRFClassifier()),
        ('AdaBoost', AdaBoostClassifier(random_state=42)),
        ('RandomForest', RandomForestClassifier(random_state=442)),
        ('ExtraTree', ExtraTreeClassifier(random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(random_state=42)),
        ('Bagging', BaggingClassifier(random_state=42)),
        ('SVC', SVC(random_state=42)),
        ('KNeighbors', KNeighborsClassifier())
    ]

    grid_params = {
        'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
        'RandomForestClassifier': {
            'n_estimators': [10, 100, 300, 1000],
            'max_depth': [3, 5, 8, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'DecisionTreeClassifier': {
            'max_depth': [3, 5, 8, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'ExtraTreesClassifier': {
            'n_estimators': [10, 100, 300, 1000],
            'max_depth': [3, 5, 8, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']  # Add criterion for potential exploration
        },
        'XGBClassifier': {
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 300, 1000],
            'max_depth': [3, 5, 8],
            'gamma': [0, 0.1, 0.5],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.5, 0.7, 1.0]
        },
        'XGBRFClassifier': {
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 300, 1000],
            'max_depth': [3, 5, 8],
            'gamma': [0, 0.1, 0.5],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.5, 0.7, 1.0]
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 1.0]
        },
        'GradientBoostingClassifier': {
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 300, 1000],
            'max_depth': [3, 5, 8],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'BaggingClassifier': {
            'n_estimators': [50, 100, 200],
            'base_estimator': DecisionTreeClassifier()  # Or other base model
            # You can further tune hyperparameters of the base model within BaggingClassifier
        },
        'SVC': {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf']  # Explore different kernels if needed
        },
        'KNeighborsClassifier': {
            'n_neighbors': list(range(1, 21)),  # Wider range for k exploration
            'weights': ['uniform', 'distance']  # Consider different weighting schemes
        }
    }

    grid_results = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    for name, model in models:
        if name in grid_params:
            # Adjust cv = 5 and scoring
            grid_search = GridSearchCV(model, grid_params[name], cv=cv, scoring=accuracy_score)
            grid_search.fit(X, y)
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
        else:
            model.fit(X, y)
            best_params = model.get_params()  # Get default parameters
            prediction = model.predict(X)
            best_score = accuracy_score(y, prediction)

        grid_results.append({'model': name, 'score': best_score, 'params': best_params})

    return pd.DataFrame(grid_results)


def random_search_classification_models(X, y, n_iter=20):  # Adjust n_iter as needed
    """
    Performs random search on various classification models and returns a DataFrame
    containing model names, scores, and best parameters.

    Args:
        X (pandas.DataFrame): Training data features.
        y (pandas.Series): Training data target labels.
        n_iter (int, optional): Number of random parameter sets to try. Defaults to 20.

    Returns:
        pandas.DataFrame: DataFrame containing model name, score, and best parameters.
    """

    models = [
        ('LogisticRegression', LogisticRegression(random_state=42)),
        ('RandomForest', RandomForestClassifier(random_state=42)),
        ('DecisionTree', DecisionTreeClassifier(random_state=42)),
        ('ExtraTrees', ExtraTreesClassifier(random_state=42)),
        ('XGBoost', XGBClassifier()),  # Adjust hyperparameters as needed
        ('XGBoostRandomForest', XGBRFClassifier()),  # Adjust hyperparameters as needed
        ('AdaBoost', AdaBoostClassifier(random_state=42)),
        ('RandomForest', RandomForestClassifier(random_state=42)),  # Explore different params
        ('ExtraTree', ExtraTreeClassifier(random_state=42)),  # Assuming typo
        ('GradientBoosting', GradientBoostingClassifier(random_state=42)),
        ('Bagging', BaggingClassifier(random_state=42)),
        ('SVC', SVC(random_state=42)),
        ('KNeighbors', KNeighborsClassifier())
    ]

    param_distributions = {
        'LogisticRegression': {'C': [10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 100],
                               'class_weight': ['balanced', {}]},  # Logarithmic for regularization
        'RandomForestClassifier': {
            'n_estimators': [10, 100, 300, 1000],  # Uniform distribution
            'max_depth': np.random.randint(1, 101, size=n_iter)  # Integer uniform
        },
        'DecisionTreeClassifier': {
            'max_depth': np.random.randint(1, 101, size=n_iter)  # Integer uniform
        },
        'ExtraTreesClassifier': {
            'n_estimators': [10, 100, 300, 1000],  # Uniform distribution
            'max_depth': np.random.randint(1, 101, size=n_iter),  # Integer uniform
            'criterion': ['gini', 'entropy']  # Uniform categorical
        },
        'XGBClassifier': {  # Assuming XGBoost is installed
            'learning_rate': stats.uniform(loc=0.01, scale=0.29),  # Uniform
            'n_estimators': np.random.randint(100, 1001, size=n_iter),  # Integer uniform
            'max_depth': np.random.randint(3, 9, size=n_iter),  # Integer uniform
            'gamma': stats.uniform(loc=0, scale=0.5),  # Uniform
            'subsample': stats.uniform(loc=0.5, scale=0.5),  # Uniform
            'colsample_bytree': stats.uniform(loc=0.5, scale=0.5)  # Uniform
        },
        'XGBRFClassifier': {  # Assuming XGBoost is installed
            'learning_rate': stats.uniform(loc=0.01, scale=0.29),  # Uniform
            'n_estimators': np.random.randint(100, 1001, size=n_iter),  # Integer uniform
            'max_depth': np.random.randint(3, 9, size=n_iter),  # Integer uniform
            'gamma': stats.uniform(loc=0, scale=0.5),  # Uniform
            'subsample': stats.uniform(loc=0.5, scale=0.5),  # Uniform
            'colsample_bytree': stats.uniform(loc=0.5, scale=0.5)  # Uniform
        },
        'AdaBoostClassifier': {
            'n_estimators': np.random.randint(50, 201, size=n_iter),  # Integer uniform
            'learning_rate': stats.uniform(loc=0.1, scale=0.9)  # Uniform
        },
        'GradientBoostingClassifier': {
            'learning_rate': stats.uniform(loc=0.01, scale=0.29),  # Uniform
            'n_estimators': np.random.randint(100, 1001, size=n_iter),  # Integer uniform
            'max_depth': np.random.randint(3, 9, size=n_iter),  # Integer uniform
            'min_samples_split': np.random.randint(2, 11, size=n_iter),  # Integer uniform
            'min_samples_leaf': np.random.randint(1, 5, size=n_iter)  # Integer uniform
        },
        'BaggingClassifier': {
            'n_estimators': np.random.randint(50, 201, size=n_iter),  # Integer uniform
            'base_estimator': DecisionTreeClassifier()  # Or other base model
            # You can further tune hyperparameters of the base model within BaggingClassifier
        },
        'SVC': {
            'C': stats.uniform(loc=0.01, scale=99.99),  # Uniform for regularization
            'kernel': ['linear', 'rbf']  # Uniform categorical
        },
        'KNeighborsClassifier': {
            'n_neighbors': np.random.randint(1, 21, size=n_iter),  # Integer uniform
            'weights': ['uniform', 'distance']  # Uniform categorical
        }
    }

    random_results = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    for name, model in models:
        if name in param_distributions:
            random_search = RandomizedSearchCV(model, param_distributions[name], cv=cv,
                                               scoring=accuracy_score, n_iter=n_iter)  # adjust cv = 5
            random_search.fit(X, y)
            best_params = random_search.best_params_
            best_score = random_search.best_score_
        else:
            model.fit(X, y)
            prediction = model.predict(X)
            best_params = model.get_params()  # Get default parameters
            best_score = accuracy_score(y, prediction)  # Assuming model has a predict method

        random_results.append({'model': name, 'score': best_score, 'params': best_params})

    return pd.DataFrame(random_results)

