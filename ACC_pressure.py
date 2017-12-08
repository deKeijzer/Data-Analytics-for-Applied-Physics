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


def pressure_plot(x, y, sample_number, x_label, y_label):
    """
    Creates a plot of the data.
    Data must be in df_TN['...'] format.
    :param x:
    :param y:
    :param sample_number:
    :param x_label:
    :param y_label:
    :return:
    """
    min_1 = 2500
    max_1 = 6000
    min_2 = 7300
    max_2 = 11850
    df_A = pd.DataFrame()  # Creates new df
    df1 = pd.DataFrame()  # Creates new df
    df2 = pd.DataFrame()
    df_A['index'] = x.index.values

    df1['x_1'] = df_A['index'].iloc[min_1:max_1]  # df for first fit
    df1['y_1'] = y.iloc[min_1:max_1]

    df2['x_2'] = df_A['index'].iloc[min_2:max_2]  # df for second fit
    df2['y_2'] = y.iloc[min_2:max_2]
    print(df1)

    sample1 = df1['y_1']
    x_bar1 = np.mean(sample1)  # x bar, sample average
    s1 = np.std(sample1)  # sigma, sample standard deviation
    delta_x1 = s1 / (len(sample1)) ** (1 / 2)  # d

    sample2 = df2['y_2']
    x_bar2 = np.mean(sample2)  # x bar, sample average
    s2 = np.std(sample2)  # sigma, sample standard deviation
    delta_x2 = s2 / (len(sample2)) ** (1 / 2)  # d

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x.index.values, y, '.', color="#ff0000", ms=5, label='Sample: pressure_' + str(sample_number))
    ax.plot(df1['x_1'], df1['y_1'], '.', color="#f9903b", ms=6, label='Data voor fit 1 : '+'$\mu_1$ = %.2f $\quad$ $\Delta \mu_1$ = %.4f' % (x_bar1, delta_x1))
    ax.plot(df2['x_2'], df2['y_2'], '.', color="#9379ff", ms=6, label='Data voor fit 2 : '+'$\mu_2$ = %.2f $\quad$ $\Delta \mu_2$ = %.4f' % (x_bar2, delta_x2))
    print(x_bar1)
    print(x_bar2)


    #plt.plot(x, 0 * x + mu, '-', color="#ff0000", ms=5) # Creates a line at (literature) y value
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    da.create_layout()
    fig.canvas.draw()


da.x_significant_digits = 6  # Number of significant digits used in plots
da.y_significant_digits = 6  # Number of significant digits used in plots

directory = 'Data\\Accelerometer (ACC)\\week4\\'
file_name = 'druk_'
sample_number = '3'
file_format = '.csv'

create_df(directory, file_name, sample_number, file_format)

pressure_plot(df['t'], df['pressure'], '3', 'Index [-]', 'Druk [mbar]')

plt.show()