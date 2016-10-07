import pandas as pd
import numpy as np
from ast import literal_eval


def read_csv(filename, split_ratio):
    read_file = pd.read_csv(filename)
    train_size = len(read_file) * split_ratio / 100
    test_size = len(read_file) - train_size
    train_data_df, test_data_df = read_file.head(int(train_size)), read_file.tail(int(test_size))
    column_drop = len(train_data_df.columns) - 3
    train_label_df = train_data_df.iloc[:, column_drop:]
    train_data_df = train_data_df.iloc[:, :column_drop]
    test_label_df = test_data_df.iloc[:, column_drop:]
    test_data_df = test_data_df.iloc[:, :column_drop]
# Convert dataframe to list
    train_data_input = np.array(train_data_df)
    train_data_output = np.array(train_label_df)
    test_data_input = np.array(test_data_df)
    test_data_output = np.array(test_label_df)
    return train_data_input, train_data_output, test_data_input, test_data_output
