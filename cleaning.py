import warnings

import numpy as np
from scipy.stats import zscore
from pyod.models.hbos import HBOS

warnings.filterwarnings("ignore")


def remove_highly_correlated_features(df, bound=0.9):
    print("Removing features with a correlation of", bound, "or greater.")
    nans = df.isnull().sum(axis=0)
    correlations = _get_top_abs_correlations(df=df.select_dtypes(include="number"))
    for index, value in correlations.items():
        firstColumn = index[0]
        secondColumn = index[1]
        if value < bound:
            break
        if (firstColumn in df.columns) & (secondColumn in df.columns):
            if nans[firstColumn] > nans[secondColumn]:
                df.drop(columns=firstColumn, inplace=True)
                print("Dropped:", firstColumn)
            else:
                df.drop(columns=secondColumn, inplace=True)
                print("Dropped:", secondColumn)
    return df


def _get_redundant_pairs(df):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def _get_top_abs_correlations(df):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = _get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr


def fill_na(df, byColumn):
    numberOfNans = df.isnull().sum(axis=1).sum()
    _ = df.groupby(byColumn, observed=True).apply(
        lambda group: group.select_dtypes(include="number").interpolate(method="index", limit_direction="both"))
    print("Filled", numberOfNans - _.isnull().sum(axis=1).sum(), "NaN-entries")
    return _.join(df.select_dtypes(include="category"), how="outer")


def drop_na_rows(df, perc=10.0):
    numberRows = df.shape[0]
    min_count = int(((100 - perc) / 100) * df.shape[1] + 1)
    df = df.dropna(axis=0, thresh=min_count)
    print("Dropped", numberRows - df.shape[0], "rows containing either", perc, "% or more than", perc, "% NaN-values")
    return df


def remove_outliers(
        df,
        density_sensitive_cols=None,
        excluded_cols=None,
        n_bins=10,
        zscore_threshold=2.5,
        verbose=False,
        contamination=0.1,
        tol=0.5,
        alpha=0.1,
):
    """
    This functions removes outliers by applying two different algorithms on specific columns:
    - outliers in density sensitive columns get detected by 'zscore'-algorthm.
    - outliers in other columns get detected by 'HBOS'-algorithm.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to be processed.
        density_sensitive_cols(list): Columns to run 'zscore'-algorithm on.
        excluded_cols(list): Columns without outlier detection.
        n_bins(int): Hyperparameter for 'HBOS'-algorithm.
        zscore_threshold(float): Hyperparameter for 'zscore'-algorithm.
        verbose(boolean): Set 'True' to get detailed logging information.
        contamination(float): Hyperparameter for 'HBOS'-algorithm.
        tol(float): Hyperparameter for 'HBOS'-algorithm.
        alpha(float): Hyperparameter for 'HBOS'-algorithm.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    if density_sensitive_cols is None:
        density_sensitive_cols = []
    if n_bins == "auto":
        n_bins = int(
            1 + 3.322 * np.log(df.shape[0])
        )  # Sturge’s Rule for detecting bin numbers automatically
    print("n_bins", n_bins)
    outlier_count = 0
    df_numeric_view = df.select_dtypes(include="number")

    for col in df_numeric_view.columns:
        if excluded_cols and col in excluded_cols:
            continue
        if col in density_sensitive_cols:
            df[f"{col}_zscore"] = np.around(np.abs(zscore(df[col])), decimals=1)
            outlier = df[df[f"{col}_zscore"] > zscore_threshold]
            outlier_count += outlier.shape[0]
            df.drop(outlier.index, inplace=True)
            if verbose:
                print(
                    f"--> {outlier.shape[0]} outlier detected and removed from {col} column using zscore"
                )
            continue
        hbos = HBOS(alpha=alpha, contamination=contamination, n_bins=n_bins, tol=tol)
        hbos.fit(df[[col]])
        df[f"{col}_anamoly_score"] = hbos.predict(df[[col]])
        outlier = df[df[f"{col}_anamoly_score"] == 1]
        outlier_count += outlier.shape[0]
        df.drop(outlier.index, inplace=True)
        if verbose:
            print(
                f"--> {outlier.shape[0]} outlier detected and removed from {col} column using HBOS algorithm"
            )

    outlier_score_cols_mask = (df.columns.str.contains("anamoly_score")) | (
        df.columns.str.contains("zscore")
    )
    df = df.loc[:, ~outlier_score_cols_mask]

    print(f"Outlier detection completed. Number of removed outlier: {outlier_count}")

    return df.reset_index(drop=True)


def remove_inorganic_stocks(df, gain=500):
    df_ = df.loc[:, ['Stock', 'Sector', 'PRICE VAR [%]']]
    top_gainers = df_[df_['PRICE VAR [%]'] >= gain]
    top_gainers.set_index("Stock", inplace=True)
    print(f'{len(top_gainers)} STOCKS with more than {gain}% gain.')
    tickers = top_gainers.index.values.tolist()
    _ = df.shape[0]
    df = df[df.Stock.isin(tickers) == False]
    print("Removed", _ - df.shape[0], "entries.")
    return df
