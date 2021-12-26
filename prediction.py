from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_data_sets_for_svm(df, test_size=0.2):
    """
    This splits a given dataframe into a train and a test set.
    ----------------------------------------------
    :param
           df (pandas.DataFrame): The given data set.
           test_size (float): Defines the size of the test set. Default is 0.2.
    """
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_index = int(len(df) * (1 - test_size))
    df_train = df[0:train_index]
    df_test = df[train_index:]
    return df_train, df_test


def make_pipeline_for_svm(cat_vars, cont_vars, model):
    """
    This builds a support vector regression pipeline based on a given list of categorical and continuous variables
    and a model.
    ----------------------------------------------
    :param
           cat_vars (list): List of categorical variables.
           cont_vars (list): List of continuous variables.
           model: Used model for prediction.
    """
    numeric_transformer = Pipeline(steps=[("standard_scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    svc_preprocessor = ColumnTransformer(
        transformers=[
            ("numerical scaler", numeric_transformer, cont_vars),
            ("one hot encoder", categorical_transformer, cat_vars),
        ]
    )
    svc_model = model
    svc_pipeline = Pipeline(
        steps=[("preprocessor", svc_preprocessor), ("svc model", svc_model)]
    )
    return svc_pipeline


def find_best_parameters_for_model(
    pipeline, X_train, y_train, model_params, scoring, n_iter, n_splits, n_repeats, verbose=True
):
    """
    This function performs a randomized grid search with five time series splits on the training set.
    ----------------------------------------------
    :param
           pipeline (sklearn.pipeline): The pipeline with the model and transformers which will be used for grid search.
           X_train (pandas.DataFrame): Training features.
           y_train (pandas.DataFrame): Target.
           model_params (dictionary): Used model parameters for the grid search.
           scoring (String): The scoring metric used for grid search.
           n_iter (int): The number of performed grid searches.
           verbose (boolean): Print detailed information while performing grid search.
    """
    print(f"Running grid search for the model based on {scoring}")
    grid_pipeline = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=model_params,
        n_jobs=1,
        n_iter=n_iter,
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42),
        scoring=scoring,
        random_state=42,
        verbose=verbose,
    )
    grid_pipeline.fit(X_train, y_train)
    print(f"Best {scoring} Score was: {grid_pipeline.best_score_}")
    print("The best hyper parameters for the model are:")
    print(grid_pipeline.best_params_)
