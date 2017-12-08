import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

global df

def create_df(directory, file_name, i, file_format):
    """
    Creates the dataframe. This function can easily be used to iterate over multiple samples.
    :param directory:
    :param file_name:
    :param i:
    :param file_format:
    :return:
    """
    global df
    path = str(directory) + str(file_name) + str(i) + str(file_format)
    if os.path.isfile(path):
        df = pd.read_csv(path, sep='\t', header=1, names=['time', 'counts'], decimal=".")
        # ---- START OF DATAFRAME MANIPULATIONS ----
        #df_TN['time'] = add_index_to_time(50)  # Adds ['time'] column to df_TN
        #df_TN['time'] = add_index_to_time(1)
    else:
        print(path+' does not excist')

