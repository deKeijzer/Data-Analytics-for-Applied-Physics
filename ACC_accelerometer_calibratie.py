import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import TISTNplot as tn
import scipy as sp
import os
from pylab import *
from scipy.stats import norm
from matplotlib import rc
import Data_analytics_TN as da

global df


def add_index_to_time(sample_rate):
    """
    Adds a time dataframe to the sample df when the sample rate (Hz) is known.
    :return:
    """
    time = []
    for i in range(len(df)):
        time.append(i/sample_rate)
    return time


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


def pressure_plot(x, y, sample_number, x_label, y_label, min_1, max_1, sample_rate):
    """
        Creates a plot of the data.
    Data must be in df_TN['...'] format.
    :param x:
    :param y:
    :param sample_number:
    :param x_label:
    :param y_label:
    :param min_1: min index
    :param max_1: max index for data selection
    :param sample_rate: in Hz
    :return:
    """
    df_A = pd.DataFrame()  # Creates new df
    df1 = pd.DataFrame()  # Creates new df
    df_A['index'] = x.index.values

    df1['x_1'] = df_A['index'].iloc[min_1:max_1]  # df for first fit
    df1['y_1'] = y.iloc[min_1:max_1]

    sample1 = df1['y_1']
    x_bar1 = np.mean(sample1)  # x bar, sample average
    s1 = np.std(sample1)  # sigma, sample standard deviation
    delta_x1 = s1 / (len(sample1)) ** (1 / 2)  # d

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Hoogte berekenen
    a_int = trapzArray(df1['y_1'], 1/sample_rate)  # eerste integraal

    #b_int = trapzArray(a_int, 1)  # tweede intergraal

    b_int = np.trapz(a_int, dx=1/sample_rate)

    ax.plot(x.index.values, y, '.',
            color="#ff0000", ms=5, label='Sample: acc_' + str(sample_number)+'$\quad $ Sample Rate: %.1f Hz' % sample_rate)
    ax.plot(df1['x_1'], df1['y_1'], '.', color="#f9903b",
            ms=6, label='Berekende hoogte : '+'$\quad h$ = %.2f m' % (b_int))

    #plt.plot(x, 0 * x + mu, '-', color="#ff0000", ms=5) # Creates a line at (literature) y value
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    da.create_layout()
    fig.canvas.draw()


def trapzArray(y, dx):
    """
    Numerically integrates an array y
    :param y:
    :param dx:
    :return:
    """
    F = []
    for i in range(len(y)):
        F.append(np.trapz (y[0:i], dx = dx))
    return F

def call_plot(x, y):
    """
        Creates a plot of the data.
    Data must be in df_TN['...'] format.
    :param x:
    :param y:
    :param sample_number:
    :param x_label:
    :param y_label:
    :param min_1: min index
    :param max_1: max index for data selection
    :param sample_rate: in Hz
    :return:
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x.index.values, y, '.',
            color="#ff0000", ms=5, label='Sample: acc_')

    plt.xlabel('Tijd [uren]')
    plt.ylabel('y')
    da.create_layout()
    fig.canvas.draw()

da.x_significant_digits = 3  # Number of significant digits used in plots
da.y_significant_digits = 3  # Number of significant digits used in plots

directory = 'Data\\Accelerometer (ACC)\\Acc calibratie\\'
file_name = 'call_'
sample_number = '1'
file_format = '.csv'

create_df(directory, file_name, sample_number, file_format)

df['t'] = df['t'] - df['t'][0]
df['t'] = df['t']/(60**2)

#call_plot(df['t'], df['acc_z'])

enable_histogram = True

sample = df['acc_z'].iloc[0:10]

x_bar = np.mean(sample) #x bar, sample average
s = np.std(sample) #sigma, sample standard deviation
delta_x = s/(len(sample))**(1/2) #d
mode = sp.stats.mode(sample, axis=None)

if enable_histogram:
    # data, number of bars, Enable gauss fit, Enable_axis_lim, x_lim, y_lim
    da.histogram(sample, 25, False, False, [-1.02, -1.0055], [0, 250],
                 r'$ \mathrm{Sample} \: 2: \quad \overline{\tau}_0 = %s \quad \Delta \overline{\tau}_0 = %s $'
                 % (da.dot_to_comma(x_bar, 3), da.dot_to_comma(delta_x, 3)),
                 r'$ \mathrm{Levensduur} \: \tau_0 \: [\mu s] $',
                 r'$ \mathrm{Frequentie} \: n \: [-] $')

#pressure_plot(df['t'], df['acc_z'], sample_number, 'Index [-]', 'Versnelling in $g$ [m/s]', 660, 890, 12.5)

plt.show()