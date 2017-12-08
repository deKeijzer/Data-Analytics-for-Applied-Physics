import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import Data_analytics_TN as da
import scipy as sp

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
        df = pd.read_csv(path, sep=',', header=None, names=['t', 'acc_x', 'acc_y', 'acc_z', 'humidity',
                            'temp_from_hum', 'temp_from_pressure',
                            'pressure', 'compass_x', 'compass_y', 'compass_z',
                            'gyro_x', 'gyro_y', 'gyro_z'], decimal=".")
        # ---- START OF DATAFRAME MANIPULATIONS ----
        #df_TN['time'] = add_index_to_time(50)  # Adds ['time'] column to df_TN
        #df_TN['time'] = add_index_to_time(1)
    else:
        print(path+' does not excist')


def add_index_array(sample_number):
    global df
    df['index'] = df.index.values
    df.to_csv('C:\\Users\\BDK\\PycharmProjects\\untitled\\Data Analytics for Applied Physics\\Data\\'
              'Accelerometer (ACC)\\week4\\slinger_modified_'+sample_number+'.csv', sep='\t')

da.x_significant_digits = 6  # Number of significant digits used in plots
da.y_significant_digits = 6  # Number of significant digits used in plots

directory = 'Data\\Accelerometer (ACC)\\week4\\'
file_name = 'slinger_'
sample_number = '2'
file_format = '.csv'

create_df(directory, file_name, sample_number, file_format)

add_index_array(sample_number)

