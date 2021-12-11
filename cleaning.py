def remove_highly_correlated_features(df, n=5):
    nans = df.isnull().sum(axis=0)
    correlations = _get_top_abs_correlations(df=df, n=n)
    for firstColumn, secondColumn in correlations.index:
        if firstColumn in df.columns & secondColumn in df.columns:
            if nans[firstColumn] > nans[secondColumn]:
                df.drop(columns=firstColumn, inplace=True)
            else:
                df.drop(columns=secondColumn, inplace=True)


def _get_redundant_pairs(df):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def _get_top_abs_correlations(df, n):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = _get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
