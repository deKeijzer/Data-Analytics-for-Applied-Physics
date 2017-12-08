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
    min_1 = 760
    max_1 = 1520
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

    sample_rate = 1/25

    # Hoogte berekenen
    a_int = trapzArray(df1['y_1'], sample_rate)  # eerste integraal

    #b_int = trapzArray(a_int, 1)  # tweede intergraal

    b_int = np.trapz(a_int, dx=sample_rate)

    ax.plot(x.index.values, y, '.', color="#ff0000", ms=5, label='Sample: acc_' + str(sample_number))
    ax.plot(df1['x_1'], df1['y_1'], '.', color="#f9903b", ms=6, label='Berekende hoogte : '+'$\quad h$ = %.2f m' % (b_int))

    #plt.plot(x, 0 * x + mu, '-', color="#ff0000", ms=5) # Creates a line at (literature) y value
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    da.create_layout()
    fig.canvas.draw()



def trapzArray(y, dx):
    F = []
    for i in range(len(y)):
        F.append(np.trapz (y[0:i], dx = dx))
    return F

da.x_significant_digits = 3  # Number of significant digits used in plots
da.y_significant_digits = 3  # Number of significant digits used in plots

directory = 'Data\\Accelerometer (ACC)\\'
file_name = 'acc_'
sample_number = '14'
file_format = '.csv'

create_df(directory, file_name, sample_number, file_format)

df['acc_z'] = df['acc_z']-0.981  # valversnelling van de gemeten versnelling halen

pressure_plot(df['t'], df['acc_z'], sample_number, 'x', 'y')

plt.show()







"""
Start of dataframe settings
"""
df_edited = pd.DataFrame()

sample_number = '2'

min_index_val = 12*10**3
max_index_val = 14*10**3
#sample = df['z'].iloc[min_index_val:max_index_val]  # The sample to do statistical analysis on
sample = df['t']
"""
End of dataframe settings
"""

"""
-----START OF USER VARIABLES-----
"""
# Label data
y_label = 'Versnelling'
x_label = 'Tijd (s)'

# Edit the df_live to df_edited using data_hist_to_raw()
#data_hist_to_raw()

#df = df_edited  # Specify which dataframe to use for calculations & plots


#df['time'] = df['time']*10E-6  # Significance corrections


# Remove 0's from df because those arn't measures values from MuonLabIII
# df = df[~(df == 0).any(axis=1)] # Verpest de fit
# print(df['time'])

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
    create_layout()

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


def numeric_integration(x, y):
    return np.trapz(y, x)


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