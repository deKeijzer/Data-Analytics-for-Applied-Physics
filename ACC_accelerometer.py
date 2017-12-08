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

da.x_significant_digits = 3  # Number of significant digits used in plots
da.y_significant_digits = 3  # Number of significant digits used in plots

directory = 'Data\\Accelerometer (ACC)\\'
file_name = 'acc_'
sample_number = '19'
file_format = '.csv'

create_df(directory, file_name, sample_number, file_format)

df['acc_z'] = df['acc_z']-0.981  # valversnelling van de gemeten versnelling halen

pressure_plot(df['t'], df['acc_z'], sample_number, 'Index [-]', 'Versnelling in $g$ [m/s]', 660, 890, 12.5)

#plt.savefig(str(directory)+'Plots\\'+'acc_'+sample_number+'.png')

a = [2.99, 2.41, 3.33, 4.52, 3.61, 3.06]
b = [8.83, 8.46, 6.67, 9.28]

df_a = pd.DataFrame()
df_b = pd.DataFrame()

df_a['a'] = a
df_b['b'] = b

print('mean verdieping delta 1: '+str(np.mean(df_a['a'])))
print('mean verdieping delta 2: '+str(np.mean(df_b['b'])))
print('delta verdieping delta 1: '+str(np.std(df_a['a'])/(len(df_a['a']))**(1/2)))
print('delta verdieping delta 1: '+str(np.std(df_b['b'])/(len(df_b['b']))**(1/2)))

plt.show()







mu = -1  # literature value to compare the sample to


"""
-----END OF USER VARIABLES-----
"""

x_bar = np.mean(sample) #x bar, sample average
s = np.std(sample) #sigma, sample standard deviation
delta_x = s/(len(sample))**(1/2) #d
mode = sp.stats.mode(sample, axis=None)

# Code needed to use LaTeX in plots
rc('font', **{'family': 'sans-serif','sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)


def create_plot_without_error(x, y, sample_number, x_label, y_label):
    """
    Creates a plot of the data.
    Data must be in df['...'] format.
    :param x:
    :param y:
    :return:
    """
    plt.plot(x, y, '.', color="#ff0000", ms=5, label='Sample '+str(sample_number))
    #plt.plot(x, 0 * x + mu, '-', color="#ff0000", ms=5) # Creates a line at (literature) y value
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    da.create_layout()

def save_multiple_plots(directory, file_name, file_format, start, stop):
    """
    Saves plots from multiple samples.
    Note that the function to use with its variables is set inside this function.
    For
    :param function:
    :param directory:
    :param file_name:
    :param file_format:
    :param range:
    :return:
    """
    global df

    for i in range(start, stop):
        path = str(directory) + str(file_name) + str(i) + str(file_format)
        if os.path.isfile(path):
            create_df(directory, file_name, i, file_format)
            create_plot_without_error(df['time'], df['acc_z'], i,
                                      r'$\mathrm{Tijd} \: [s]$',
                                      r'$ \mathrm{Versnelling in \:} g \: [ms^{-2}]$')
            plt.savefig(str(directory)+'Plots\\'+'acc_'+str(i)+'.png')
            plt.clf()

print('x_bar: %s'% x_bar)
print('s: %s'% s)
print('delta_x: %s'% delta_x)
da.t_test(sample, mu)

#print(stats.norm.ppf(1.96))
#print(stats.norm.ppf(1.96))


#create_plot(df['U'], df['delta U'], df['mgd'], df['delta mgd'])
#create_plot_without_error(df['t'], df['pressure'], sample_number, 'x', 'y')
#linear_regression_plot(df['U'], df['delta U'], df['mgd'], df['delta mgd'])
#logarithmic_fit_plot(df['time'], df['counts'])
#line_fit_plot(df['time'], df['z'])
#df_to_csv('raw.scv') #saves df to scv file 'file name'
#logarithmic_fit_plot(df['time'], df['counts'])

#acc_line_fit_plot(df['time'], df['z'], sample_number, r'$\mathrm{Tijd \:} (s)$',
#                  r'$ \mathrm{Versnelling \: in \:} g \: (ms^-2)}$')

#multi_plot_sense_hat()

#plt.show()

#print(numeric_integration(df['time'], df['z']))

#save_multiple_plots('Data\\Accelerometer (ACC)\\', 'acc_', '.csv', 1, 21)

"""
TODO:
Add a chee for the linear regression plot
"""