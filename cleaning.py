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
