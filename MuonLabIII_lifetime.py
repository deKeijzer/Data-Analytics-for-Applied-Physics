import pandas as pd
import numpy as np
import Data_analytics_TN as da
import os
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import rc
import locale

global df

plt.rcdefaults()
# Set to dutch locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "dutch")
# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True

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


sample_name = 'time'  # df[sample_name], sample_name for manipulations

da.min_index_val = 12*10**3
da.ax_index_val = 14*10**3
"""
Start of dataframe settings
"""
df_edited = pd.DataFrame()

folder_number = '15'
sample_number = ''
path = 'Data\\MuonLab\\'+folder_number+'\\'
create_df(path, 'lifetime', sample_number, '.txt')
# Remove 0's from df because those arn't measures values from MuonLabIII
df = df[~(df == 0).any(axis=1)]  # Verpest de fit
da.df_TN = df


convert_hist_to_raw = True

if convert_hist_to_raw:
    da.data_hist_to_raw(sample_name, 'counts')
    df = da.df_TN_hist_to_raw
    df = df.sort_values(sample_name, ascending=False)  # Sort df['lifetimes'] from high to low value
    df = df.reset_index(drop=True)  # Reset index numbers back to default (0...100) instead of (100...0)

#sample = df['z'].iloc[min_index_val:max_index_val]  # The sample range to do statistical analysis on
sample = df[sample_name]  # df to do statistics on
da.sample = sample

"""
End of dataframe settings
"""

"""
-----START OF USER VARIABLES-----
"""
# Label data
da.x_label = r'$ \mathrm{Tijd} \: [\mu s] $'
da.y_label = r'$ \mathrm{Hit \: nummer} \: [-] $'

da.mu = 2.19704  # literature value to compare the sample to
da.x_significant_digits = 3  # Number of significant digits used in plots
da.y_significant_digits = 3  # Number of significant digits used in plots

"""
-----END OF USER VARIABLES-----
"""

da.x_bar = np.mean(sample) # x bar, sample average
da.s = np.std(sample) # sigma, sample standard deviation
da.delta_x = da.s/(len(sample))**(1/2) #d
da.mode = sp.stats.mode(sample, axis=None)

print('x_bar: %s'% da.x_bar)
print('s: %s'% da.s)
print('delta_x: %s'% da.delta_x)
da.t_test(da.sample, da.mu)


def create_expected_data_array(observed_array, a, b, c):
    expected_array = []  # array of expected data created from fit
    for i in range(len(observed_array)):
        expected_i = da.exponential_decay_fit(i, a, b, c)
        expected_array.append(expected_i)
    df['expected'] = expected_array

def add_error(sample_array, error_value):
    error_array = []  # array of expected data created from fit
    for i in range(len(sample_array)):
        error_array.append(error_value)
    df['error'] = error_array


add_error(df['time'], 10*10**-9)

#create_expected_data_array(df['time'], 383, 2.25, 0)

enable_fit_plot = False
enable_histogram = False

if enable_fit_plot:
    da.create_plot_without_error(df['time'], df.index.values, '2', da.x_label, da.y_label)
    popt, pcov = sp.optimize.curve_fit(da.exponential_decay_fit, df['time'], df.index.values, bounds=(0, [500., 3., 3]))
    plt.rc('text', usetex=True)
    plt.plot(df['time'], da.exponential_decay_fit(df['time'], *popt), 'r-', color='#0000ff',
             label=r"$ N_0 \cdot e^{-\frac{t}{\tau_0}}+c "
                   r"\quad N_0=%5.0f \quad \tau_0=%5.2f$ \quad \text{c}=%5.2f$ $" % tuple(popt))
    #plt.plot(df['expected'], df.index.values, color="#000000")
    da.create_layout()
    plt.grid()

if enable_histogram:
    # data, number of bars, Enable gauss fit, Enable_axis_lim, x_lim, y_lim
    da.histogram(sample, 100, False, False, [-1.02, -1.0055], [0, 250],
                 r'$ \mathrm{Sample} \: 2: \quad \overline{\tau}_0 = %s \quad \Delta \overline{\tau}_0 = %s $'
                 % (da.dot_to_comma(da.x_bar, 3), da.dot_to_comma(da.delta_x, 3)),
                 r'$ \mathrm{Levensduur} \: \tau_0 \: [\mu s] $',
                 r'$ \mathrm{Frequentie} \: n \: [-] $')



#print(df['time'])

#plt.show()

df.to_csv('C:\\Users\\BDK\\PycharmProjects\\untitled\\Data Analytics for Applied Physics'
          '\\Data\\MuonLab\\'+folder_number+'\\lifetime_raw.txt', sep='\t')




