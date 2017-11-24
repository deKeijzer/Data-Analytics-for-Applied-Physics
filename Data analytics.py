import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import TISTNplot as tn
from pylab import *
import scipy as sp
from scipy.stats import norm
from matplotlib import rc

"""
Start of dataframe settings
"""
df_live = pd.read_csv('data\Acc meter\meting 8.csv'
                 , sep=',', header=1, names=['x', 'y', 'z'],
                 decimal=".")
df_edited = pd.DataFrame()
"""
End of dataframe settings
"""

def data_hist_to_raw(): # 1e meetwaarde lijkt niet te worden meegenomen?
    """
    Converts histogram data to raw data.
    The required data format is ['x value', 'counts']
    :param time:
    :param counts:
    :return:
    """
    a = 'time'
    b = 'counts'

    df_copy = df_live.copy()
    time = []
    index = []
    for i in range(len(df_copy)):
        val = df_copy.ix[i][a]
        for j in range(0, int(abs(df_copy.ix[i][b]))):
            time.append(val)
    for k in range(len(time)):
        index.append(k+1)
    df_edited['time'] = time  # Creates ['time'] column containing the time array data
    df_edited['index'] = index


def add_index_to_time(sample_rate):
    """
    Adds a time dataframe to the sample df when the sample rate (Hz) is known.
    :return:
    """
    time = []
    for i in range(len(df)):
        time.append(i/sample_rate)
    return time


"""
-----START OF USER VARIABLES-----
"""
# Label data
y_label = 'Versnelling'
x_label = 'Tijd (s)'

# Edit the df_live to df_edited using data_hist_to_raw()
# data_hist_to_raw()


df = df_live  # Specify which dataframe to use for calculations & plots


#df['time'] = df['time']*10E-6  # Significance corrections
df['time'] = add_index_to_time(50)  # Adds ['time'] column to df

# Remove 0's from df because those arn't measures values from MuonLabIII
# df = df[~(df == 0).any(axis=1)] # Verpest de fit
# print(df['time'])

sample = df['time']  # The sample to do statistical analysis on
mu = 6  # literature value to compare the sample to
x_significant_digits = 3  # Number of significant digits used in plots
y_significant_digits = 3  # Number of significant digits used in plots

"""
-----END OF USER VARIABLES-----
"""

x_bar = np.mean(sample) #x bar, sample average
s = np.std(sample) #sigma, sample standard deviation
delta_x = s/(len(sample))**(1/2) #d
mode = sp.stats.mode(sample, axis=None)

# Code needed to use LaTeX in plots
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def histogram(data, num_bins, enable_gauss_fit, enable_axis_lim, x_lim, y_lim): # even check of sigma wel s is
    """
    Plots a histogram with number_of_bins of given data.
    :param data:
    :param number_of_bins:
    :return:
    """
    #number_of_bins = 2*st.iqr(data)/(len(data)**(1/3)) # Freedman–Diaconis rule, werkt niet bij lage getallen
    fig = plt.figure()

    if enable_axis_lim:
        axes = plt.gca()
        axes.set_xlim(x_lim)
        axes.set_ylim(y_lim)

    # example data
    x = sample

    # Histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='pink', alpha=0.5, ec='black', label='Sample')

    # Creates a best gauss fit line
    if enable_gauss_fit:
        y = mlab.normpdf(bins, x_bar, s)
        plt.plot(bins, y, 'r--', color="#ff0000", label=r'$\mathrm{}\ \mu=%.2E,\ \sigma=%.2E$'% (x_bar, s))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Puts the legend in the best location
    plt.subplots_adjust(left=0.15)
    plt.legend(loc='best')
    plt.show()


def t_test(sample, mu):
    """
    Calculates the T-test for the mean of ONE group of scores.
    This is a two-sided test for the null hypothesis that the expected value (mean)
    of a sample of independent observations a is equal to the given population mean, mu.

    :param sample:
    :param mu:
    :return:
    """
    ttest = sp.stats.ttest_1samp(sample, mu)
    t = ttest[0]
    p = ttest[1]
    p_percent = p*100
    to_reject = 'NO'
    if p_percent < 5:
        to_reject = 'YES'
    result =\
        't test: '' \
        ''\n \t \t t = %s  '' \
        ''\n \t \t p = %s '' \
        ''\n \t \t p(%%) = %s'\
        '\n \t \t Reject the null hypothesis: %s' % (t, p, p_percent, to_reject)

    return print(result)

def create_layout():
    """
    Creates the plot layout as set by certain TIS-TN standards.
    """
    tn.PRECISION_X = x_significant_digits
    tn.PRECISION_Y = y_significant_digits
    tn.fix_axis(plt.gca())
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid()


def create_plot(x, x_error, y1, y1_error):
    """
    Creates a plot of the data.
    Data must be in df['...'] format.
    :param x:
    :param y:
    :return:
    """
    plt.errorbar(x, y1, yerr=y1_error, xerr=x_error, capsize=5, fmt='.', color="#0000ff", ms=5)
    #plt.plot(x, 0 * x + mu, '-', color="#ff0000", ms=5) # Creates a line at (literature) y value
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    create_layout()
    plt.show()

def create_plot_without_error(x, y1):
    """
    Creates a plot of the data.
    Data must be in df['...'] format.
    :param x:
    :param y:
    :return:
    """
    plt.errorbar(x, y1, capsize=5, fmt='.', color="#0000ff", ms=5)
    #plt.plot(x, 0 * x + mu, '-', color="#ff0000", ms=5) # Creates a line at (literature) y value
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    create_layout()
    plt.show()


def linear_regression_plot(x, x_error, y, y_error):
    """
    Creates a plot of the data, includes a fitted line through linear regression.
    :param x:
    :param x_error:
    :param y:
    :param y_error:
    :return:
    """
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 14,
            }

    a, b = polyfit(x, y, 1)
    plt.plot(x, a * x + b, '--k', label=r'$\mathrm{y=%.2Ex+%.2E}\ $'% (a, b))  # Plots the fitted line
    plt.errorbar(x, y, yerr=y_error, xerr=x_error, capsize=5, fmt='.', color='#0000ff', ms=10)  # Plots the data points
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    create_layout()
    plt.show()

def exponential_fit(x, a, b, c):
    """
    Logarithmic fit
    :param x:
    :param a:
    :param c:
    :return:
    """
    return (1/a)*np.exp(-x/b)+c

def logarithmic_fit_plot(x, y): # WIP
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)
    xdata = x
    ydata = y
    plt.rc('text', usetex=True)
    plt.plot(xdata, ydata, '.', label='sample')
    popt, pcov = sp.optimize.curve_fit(exponential_fit, xdata, ydata, bounds=(0, [3., 3., 3]))
    plt.plot(xdata, exponential_fit(xdata, *popt), 'r-',
             #label=r"$\frac{1}{\tau_0}e^{\frac{-x}{\tau_0}}, \tau_0=%5.3f, b=%5.3f$, c=%5.3f$ $" % tuple(popt))
             label=r"$\frac{1}{a}e^{\frac{-x}{b}}+c, a=%5.3f, b=%5.3f$, c=%5.3f$ $" % tuple(popt))
    plt.legend()
    create_layout()
    plt.show()

def df_to_csv(file_name):
    df.to_csv(file_name, sep='\t')

print('x_bar: %s'% x_bar)
print('s: %s'% s)
print('delta_x: %s'% delta_x)
t_test(sample, mu)

#print(stats.norm.ppf(1.96))
#print(stats.norm.ppf(1.96))
#histogram(sample, 25, False, False, [0, 1], [0,25]) # data, number of bars, Enable gauss fit, Enable_axis_lim, x_lim, y_lim
#create_plot(df['U'], df['delta U'], df['mgd'], df['delta mgd'])
create_plot_without_error(df['time'], df['z'])
#linear_regression_plot(df['U'], df['delta U'], df['mgd'], df['delta mgd'])
#logarithmic_fit_plot(df['time'], df['counts'])
#df_to_csv('raw.scv') #saves df to scv file 'file name'
#logarithmic_fit_plot(df['time'], df['counts'])

"""
TODO:
Add a chee for the linear regression plot
"""