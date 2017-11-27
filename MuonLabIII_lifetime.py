import pandas as pd
import numpy as np
import Data_analytics_TN as da
import os
import scipy as sp
import matplotlib.pyplot as plt
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
df = df[~(df == 0).any(axis=1)]
da.df_TN = df

da.data_hist_to_raw('lifetimes', 'counts')
df = da.df_TN_hist_to_raw
df = df.sort_values('lifetimes', ascending=False)  # Sort df['lifetimes'] from high to low value
df = df.reset_index(drop=True)  # Reset index numbers back to default (0...100) instead of (100...0)

sample = df['lifetimes']
da.sample = sample

"""
End of dataframe settings
"""

"""
-----START OF USER VARIABLES-----
"""
da.y_label = 'Versnelling'
da.x_label = 'Tijd (s)'

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

# data, number of bars, Enable gauss fit, Enable_axis_lim, x_lim, y_lim
da.histogram(sample, 100, False, False, [-1.02, -1.0055], [0, 250],
             r'$ \mathrm{Sample} \: 2: \quad \overline{\tau}_0 = %s \quad \Delta \overline{\tau}_0 = %s $'
             % (da.dot_to_comma(da.x_bar, 3), da.dot_to_comma(da.delta_x, 3)),
             r'$ \mathrm{Levensduur} \: \tau_0 \: [\mu s] $',
             r'$ \mathrm{Frequentie} \: n \: [-] $')


plt.show()

