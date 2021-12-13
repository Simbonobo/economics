import os
import warnings

import pandas as pd

from datetime import datetime

warnings.filterwarnings("ignore")


def _get_data_path():
    """
    This function finds the path of data folder in the current project.
    ----------------------------------------------
    :return:
        String: the data folder path
    :raises
        FileNotFoundError: if the data directory can't be located.
    """
    if os.path.isdir(os.path.join(os.getcwd(), "data")):
        path = os.path.join(os.getcwd(), "data")
    elif os.path.isdir(os.path.join(os.getcwd(), "..", "data")):
        path = os.path.join(os.getcwd(), "..", "data")
    else:
        raise FileNotFoundError("The data folder could not be found")

    return path


def read_csv(path=_get_data_path(), years=None):
    if years is None:
        file = "Cleaned_Financial_Data.csv"
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, index_col=0)

        return df

    for year in years:
        file = year + "_Financial_Data.csv"
        file_path = os.path.join(path, file)
        _ = pd.read_csv(file_path, index_col=0)
        _.rename(columns={_.columns[-2]: "PRICE VAR [%]"}, inplace=True)
        _["Year"] = datetime.strptime(year, '%Y').year
        if year is years[0]:
            df = _
        else:
            df = pd.concat([df, _], sort=False).drop_duplicates()
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Stock"}, inplace=True)
    df.sort_values(by=["Stock", "Year"], inplace=True)

    return df


def write_csv(df, filename, path=_get_data_path()):
    path = os.path.join(path, filename)
    df.to_csv(path_or_buf=path)
