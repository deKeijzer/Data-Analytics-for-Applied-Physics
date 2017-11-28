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
import locale

global df_TN, x_bar, x_significant_digits, y_significant_digits, s, x_label, y_label, df_TN_hist_to_raw

# Code required to use LaTeX in plots
rc('font', **{'family': 'sans-serif','sans-serif': ['Helvetica']})
rc('text', usetex=True)

plt.rcdefaults()
# Set to dutch locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "dutch")
# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True


def data_hist_to_raw(x_value, counts):  # 1e meetwaarde lijkt niet te worden meegenomen?
    """
    Converts histogram data to raw data.
    The required data format is ['x_value', 'counts'].
    :param x_value: df['x_value']
    :param counts:  df['counts']
    :return:
    """
    global df_TN, df_TN_hist_to_raw
    a = x_value
    b = counts

    df_TN_copy = df_TN.copy()
    time = []
    for i in range(len(df_TN_copy)):
        val = df_TN_copy.iloc[i][a]
        for j in range(0, int(abs(df_TN_copy.iloc[i][b]))):
            time.append(val)
    d = {x_value: time}
    df_TN_hist_to_raw = pd.DataFrame(data=d)


def add_index_to_time(sample_rate):
    """
    Adds a time dataframe to the sample df_TN when the sample rate (Hz) is known.
    :return:
    """
    time = []
    for i in range(len(df_TN)):
        time.append(i/sample_rate)
    return time


def create_df_TN(directory, file_name, i, file_format):
    """
    Creates the dataframe. This function can easily be used to iterate over multiple samples.
    :param directory:
    :param file_name:
    :param i:
    :param file_format:
    :return:
    """
    global df_TN
    path = str(directory) + str(file_name) + str(i) + str(file_format)
    if os.path.isfile(path):
        df_TN = pd.read_csv(path, sep=',', header=None, names=['lifetime', 'counts'], decimal=".")
        # ---- START OF DATAFRAME MANIPULATIONS ----
        #df_TN['time'] = add_index_to_time(50)  # Adds ['time'] column to df_TN
        #df_TN['time'] = add_index_to_time(1)
    else:
        print(path+' does not excist')


def histogram(data, num_bins, enable_gauss_fit, enable_axis_lim, x_lim, y_lim, sample_name, x_label, y_label):
    # even check of sigma wel s is
    """
    Plots a histogram with number_of_bins of given data.
    :param data:
    :param num_bins:
    :param enable_gauss_fit:
    :param enable_axis_lim:
    :param x_lim:
    :param y_lim:
    :param sample_name:
    :param x_label:
    :param y_label:
    :return:
    """
    global x_bar, s
    #number_of_bins = 2*st.iqr(data)/(len(data)**(1/3)) # Freedman–Diaconis rule, werkt niet bij lage getallen
    fig = plt.figure()

    if enable_axis_lim:
        axes = plt.gca()
        axes.set_xlim(x_lim)
        axes.set_ylim(y_lim)

    # example data
    x = sample

    # Histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=False, facecolor='pink', alpha=0.5, ec='black', label=sample_name)

    # Creates a best gauss fit line
    if enable_gauss_fit:
        y = mlab.normpdf(bins, x_bar, s)
        plt.plot(bins, y, 'r--', color="#ff0000",
                 label=r'$\mathrm{}\ \overline{\mu}=%.2f,\ \sigma=%.2f$' % (x_bar, s))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Puts the legend in the best location
    plt.subplots_adjust(left=0.15)
    plt.legend(loc='best')


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
    global x_significant_digits, y_significant_digits
    tn.PRECISION_X = x_significant_digits
    tn.PRECISION_Y = y_significant_digits
    tn.fix_axis(plt.gca())
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid()


def create_plot(x, x_error, y1, y1_error):
    """
    Creates a plot of the data.
    Data must be in df_TN['...'] format.
    :param x:
    :param x_error:
    :param y1:
    :param y1_error:
    :return:
    """
    global x_label, y_label
    plt.errorbar(x, y1, yerr=y1_error, xerr=x_error, capsize=5, fmt='.', color="#0000ff", ms=5)
    #plt.plot(x, 0 * x + mu, '-', color="#ff0000", ms=5) # Creates a line at (literature) y value
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    create_layout()


def create_plot_without_error(x, y, sample_number, x_label, y_label):
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
    plt.plot(x, y, '.', color="#ff0000", ms=5, label='Sample '+str(sample_number))
    #plt.plot(x, 0 * x + mu, '-', color="#ff0000", ms=5) # Creates a line at (literature) y value
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    create_layout()


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


def exponential_decay_fit(x, a, b, c):
    """
    Plots 1/tau for lambda
    https://en.wikipedia.org/wiki/Exponential_decay
    :param x:
    :param a:
    :param c:
    :return:
    """
    return (a)*np.exp(-x/b)+c


def line_fit(x, a, b):
    """
    Logarithmic fit
    :param x:
    :param a:
    :param b:
    :return:
    """
    return a+b*x


def logarithmic_fit_plot(x, y): # WIP
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)
    xdata = x
    ydata = y
    plt.rc('text', usetex=True)
    plt.plot(xdata, ydata, '.', label='sample')
    popt, pcov = sp.optimize.curve_fit(exponential_decay_fit, xdata, ydata, bounds=(0, [3., 3., 3]))
    plt.plot(xdata, exponential_decay_fit(xdata, *popt), 'r-',
             # label=r"$\frac{1}{\tau_0}e^{\frac{-x}{\tau_0}}, \tau_0=%5.3f, b=%5.3f$, c=%5.3f$ $" % tuple(popt))
             label=r"$\frac{1}{a}e^{\frac{-x}{b}}+c, a=%5.3f, b=%5.3f$, c=%5.3f$ $" % tuple(popt))
    plt.legend()
    create_layout()


def line_fit_plot(x, y):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)
    xdata = x
    ydata = y
    plt.rc('text', usetex=True)
    plt.plot(xdata, ydata, '.', label='sample')
    popt, pcov = sp.optimize.curve_fit(line_fit, xdata, ydata, bounds=(0, [3., 3., 3]))
    plt.plot(xdata, line_fit(xdata, *popt), 'r-',
             # label=r"$\frac{1}{\tau_0}e^{\frac{-x}{\tau_0}}, \tau_0=%5.3f, b=%5.3f$, c=%5.3f$ $" % tuple(popt))
             label=r"$\frac{1}{a}e^{\frac{-x}{b}}+c, a=%5.3f, b=%5.3f$, c=%5.3f$ $" % tuple(popt))
    plt.legend()
    create_layout()


def df_TN_to_csv(file_name):
    df_TN.to_csv(file_name, sep='\t')


def save_multiple_plots(directory, file_name, file_format, start, stop):
    """
    Saves plots from multiple samples.
    Note that the function to use with its variables is set inside this function.
    :param directory:
    :param file_name:
    :param file_format:
    :param start:
    :param stop:
    :return:
    """
    global df_TN

    for i in range(start, stop):
        path = str(directory) + str(file_name) + str(i) + str(file_format)
        if os.path.isfile(path):
            create_df_TN(directory, file_name, i, file_format)
            create_plot_without_error(df_TN['time'], df_TN['acc_z'], i,
                                      r'$\mathrm{Tijd} \: [s]$',
                                      r'$ \mathrm{Versnelling in \:} g \: [ms^{-2}]$')
            plt.savefig(str(directory)+'Plots\\'+'acc_'+str(i)+'.png')
            plt.clf()


def numeric_integration(x, y):
    return np.trapz(y, x)


def acc_line_fit_plot(x, y, sample_number, x_label, y_label): # WIP
    """

    :param x:
    :param y:
    :param sample_number:
    :param x_label:
    :param y_label:
    :return:
    """
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)
    xdata = x
    ydata = y
    plt.rc('text', usetex=True)
    plt.plot(xdata, ydata, '.', color="#ff0000", label='Sample '+str(sample_number))
    min_line_1 =4*10**3
    max_line_1 =8*10**3
    min_line_2 =12*10**3
    max_line_2 =14*10**3
    popt1, pcov1 = sp.optimize.curve_fit(line_fit,
                                         xdata.iloc[min_line_1:max_line_1], ydata.iloc[min_line_1:max_line_1],
                                         bounds=([-10., -10.], [10., 10.]))
    plt.plot(xdata, line_fit(xdata, *popt1), 'r-', color="#0000ff",
             label=r"$+1g: \: a+bx, \: a=%5.3f, b=%5.3f $" % tuple(popt1))
    popt2, pcov2 = sp.optimize.curve_fit(line_fit,
                                         xdata.iloc[min_line_2:max_line_2], ydata.iloc[min_line_2:max_line_2],
                                         bounds=([-2., -2.], [2., 2.]))
    plt.plot(xdata, line_fit(xdata, *popt2), 'r-', color="#0000ff",
             label=r"$-1g: \: a+bx, \: a=%5.3f, b=%5.3f $" % tuple(popt2))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    create_layout()


def multi_plot_sense_hat():
    t = df_TN['t']
    x1 = df_TN['acc_x']
    x2 = df_TN['acc_y']

    plt.subplot(2, 1, 1)  # nrows, ncols, plot_number
    plt.plot(t, x1)
    plt.ylabel(str(x1))

    plt.subplot(2, 1, 1)  # nrows, ncols, plot_number
    plt.plot(t, x2)
    plt.ylabel(str(x2))


def dot_to_comma(number, significance):
    """
    Takes a number and replaces it's dot with a comma.
    Returns the number as a string with set significance.
    :param number:
    :return:
    """
    numb = str(number)[0:(significance+1)]

    return numb.replace('.', ',')


"""
TODO:
Add a chee for the linear regression plot
"""