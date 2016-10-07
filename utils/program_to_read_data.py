import pandas as pd
from ast import literal_eval
import numpy as np


def convert_df_tolist(*input_data):
    """
    @author: Prathyush SP

    Convert Dataframe to List

    ..todo::
        Prathyush SP; check list logic
    :param input_data: Input Data (*args)
    :return: Dataframe
    """
    # todo Prathyush SP: Perform Dataframe Validation
    dataframes = []
    for df in input_data:
        if isinstance(df, pd.DataFrame):
            dataframes.append(df.values.tolist())
        elif isinstance(df, pd.Series):
            df_list = df.to_frame().values.tolist()
            if isinstance(df_list, list):
                if isinstance(df_list[0][0], list):
                    dataframes.append([i[0] for i in df.to_frame().values.tolist()])
                else:
                    dataframes.append(df.to_frame().values.tolist())
            else:
                dataframes.append(df.to_frame().values.tolist())
    return dataframes


def read_csv(filename, split_ratio, delimiter=',', normalize=False, dtype=None, header=None, skiprows=None,
             index_col=False, label_end=True, randomize=False, return_as_dataframe=False, describe=False,
             label_vector=False):
    """
    @author: Prathyush SP

    The function is used to read a csv file with a specified delimiter

    :param filename: File name with absolute path
    :param split_ratio: Ratio used to split data into train and test
    :param delimiter: Delimiter used to split columns
    :param normalize: Normalize the Data
    :param dtype: Data Format
    :param header: Column Header
    :param skiprows: Skip specified number of rows
    :param index_col: Index Column
    :param label_end: Column which specifies result for each sample
    :param randomize: Randomize data
    :param return_as_dataframe: Returns as a dataframes
    :param describe: Describe Input Data
    :return: return train_data, train_label, test_data, test_label based on return_as_dataframe
    """
    print("Reached data")
    df = pd.read_csv(filename, sep=delimiter, index_col=index_col, header=header, dtype=dtype, skiprows=skiprows)
    if describe:
        print(df.describe())
    df = df.sample(frac=1) if randomize else df
    df = df.apply(lambda x: np.log(x)) if normalize else df
    train_size = len(df) * split_ratio / 100
    test_size = len(df) - train_size
    train_data_df, test_data_df = df.head(int(train_size)), df.tail(int(test_size))
    column_drop = len(train_data_df.columns) - 1 if label_end else (1 if index_col else 0)
    if header is None:
        train_label_df = train_data_df[column_drop].apply(literal_eval) if label_vector else train_data_df[column_drop]
        train_data_df = train_data_df.drop(column_drop, axis=1)
        test_label_df = test_data_df[column_drop]
        test_data_df = test_data_df.drop(column_drop, axis=1)
    else:
        train_label_df = train_data_df[[column_drop]]
        train_data_df = train_data_df.drop(df.columns[column_drop], axis=1)
        test_label_df = test_data_df[[column_drop]]
        test_data_df = test_data_df.drop(df.columns[column_drop], axis=1)
    if return_as_dataframe:
        return train_data_df, train_label_df, test_data_df, test_label_df
    else:
        return convert_df_tolist(train_data_df, train_label_df, test_data_df, test_label_df)
