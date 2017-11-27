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
        df = pd.read_csv(path, sep='\t', header=1, names=['lifetimes', 'counts'], decimal=".")
        # ---- START OF DATAFRAME MANIPULATIONS ----
        #df_TN['time'] = add_index_to_time(50)  # Adds ['time'] column to df_TN
        #df_TN['time'] = add_index_to_time(1)
    else:
        print(path+' does not excist')


da.min_index_val = 12*10**3
da.ax_index_val = 14*10**3
"""
Start of dataframe settings
"""
df_edited = pd.DataFrame()

sample_number = ''
create_df('Data\\MuonLab\\2\\', 'lifetime', sample_number, '.txt')
# Remove 0's from df because those arn't measures values from MuonLabIII
df = df[~(df == 0).any(axis=1)]  # Verpest de fit
da.df_TN = df

da.data_hist_to_raw('lifetimes', 'counts')
df = da.df_TN_hist_to_raw
df = df.sort_values('lifetimes', ascending=False)  # Sort df['lifetimes'] from high to low value
df = df.reset_index(drop=True)  # Reset index numbers back to default (0...100) instead of (100...0)

#sample = df['z'].iloc[min_index_val:max_index_val]  # The sample range to do statistical analysis on
sample = df['lifetimes']
da.sample = sample

"""
End of dataframe settings
"""

"""
-----START OF USER VARIABLES-----
"""
# Label data
da.y_label = 'Versnelling'
da.x_label = 'Tijd (s)'




# Edit the df_live to df_edited using data_hist_to_raw()


#df = df_edited  # Specify which dataframe to use for calculations & plots


#df['time'] = df['time']*10E-6  # Significance corrections



# print(df['time'])

da.mu = 2.19704  # literature value to compare the sample to
da.x_significant_digits = 3  # Number of significant digits used in plots
da.y_significant_digits = 3  # Number of significant digits used in plots

"""
-----END OF USER VARIABLES-----
"""

da.x_bar = np.mean(sample) #x bar, sample average
da.s = np.std(sample) #sigma, sample standard deviation
da.delta_x = da.s/(len(sample))**(1/2) #d
da.mode = sp.stats.mode(sample, axis=None)

print('x_bar: %s'% da.x_bar)
print('s: %s'% da.s)
print('delta_x: %s'% da.delta_x)
da.t_test(da.sample, da.mu)

"""
MuonLab specific code
"""
def plot_histogram_with_fit():
    # data, number of bars, Enable gauss fit, Enable_axis_lim, x_lim, y_lim
    da.histogram(sample, 100, False, False, [-1.02, -1.0055], [0, 250],
                 r'$ \mathrm{Sample} \: 2: \: \overline{\tau}_0 = %.2f , \: \Delta \overline{\tau}_0 = %.2f $'
                 % (da.x_bar, da.delta_x),
                 r'$ \mathrm{Levensduur} \: \tau_0 \: [\mu s] $',
                 r'$ \mathrm{Frequentie} \: n \: [-] $')


plot_histogram_with_fit()

#print(stats.norm.ppf(1.96))
#print(stats.norm.ppf(1.96))


#da.histogram(sample, 100, False, False, [-1.02, -1.0055], [0, 250],
#          r'$ \mathrm{Sample} \: 2: \: \overline{\tau}_0 = %.2f , \: \Delta \overline{\tau}_0 = %.2f $'
#             % (da.x_bar, da.delta_x),
#          r'$ \mathrm{Levensduur} \: \tau_0 \: [\mu s] $',
#          r'$ \mathrm{Frequentie} \: n \: [-] $') # data, number of bars, Enable gauss fit, Enable_axis_lim, x_lim, y_lim

#create_plot(df['U'], df['delta U'], df['mgd'], df['delta mgd'])
#create_plot_without_error(df['t'], df['pressure'], sample_number, 'x', 'y')
#linear_regression_plot(df['U'], df['delta U'], df['mgd'], df['delta mgd'])
#da.logarithmic_fit_plot(df['lifetimes'], df['counts'])  # df.index.values gives index numbers as an array
#line_fit_plot(df['time'], df['z'])
#df_to_csv('raw.scv') #saves df to scv file 'file name'

#acc_line_fit_plot(df['time'], df['z'], sample_number, r'$\mathrm{Tijd \:} (s)$',
#                  r'$ \mathrm{Versnelling \: in \:} g \: (ms^-2)}$')

#multi_plot_sense_hat()
print(df)
plt.show()

#print(numeric_integration(df['time'], df['z']))

#save_multiple_plots('Data\\Accelerometer (ACC)\\', 'acc_', '.csv', 1, 21)

