def remove_highly_correlated_features(df, bound=0.9):
    print("Removing features with a correlation of", bound, "or greater.")
    nans = df.isnull().sum(axis=0)
    correlations = _get_top_abs_correlations(df=df)
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
