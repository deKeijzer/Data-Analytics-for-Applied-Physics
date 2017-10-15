import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import TISTNplot as tn
from pylab import *
from scipy import stats
from scipy.stats import norm

"""
-----START OF USER VARIABLES-----
"""

df = pd.read_csv('q_berekend'\
                 , sep='\t', header=1, names=['U', 'delta U', 'x begin', 'x eind', 't',
                                              'mgd', 'delta mgd', 'q'],
                 decimal=".")

sample = df['q']  # The sample to do statistical analysis on
mu = 1.6E-19  # literature value to compare the sample to
x_significant_digits = 3  # Number of significant digits used in plots
y_significant_digits = 3  # Number of significant digits used in plots

"""
-----END OF USER VARIABLES-----
"""

x_bar = np.mean(sample) #x bar, sample average
s = np.std(sample) #sigma, sample standard deviation
delta_x = s/(len(sample))**(1/2) #d
mode = stats.mode(sample, axis=None)


def histogram(data, num_bins): # even check of sigma wel s is
    """
    Plots a histogram with number_of_bins of given data.
    :param data:
    :param number_of_bins:
    :return:
    """
    #number_of_bins = 2*st.iqr(data)/(len(data)**(1/3)) # Freedman–Diaconis rule, werkt niet bij lage getallen
    fig = plt.figure()
    # example data
    x = sample

    # Histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='pink', alpha=0.5, ec='black', label='Sample')
    # Creates a best fit line
    y = mlab.normpdf(bins, x_bar, s)
    plt.plot(bins, y, 'r--', color="#ff0000", label=r'$\mathrm{}\ \mu=%.2E,\ \sigma=%.2E$'% (x_bar, s))
    plt.xlabel('x')
    plt.ylabel('Probability')

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
    ttest = stats.ttest_1samp(sample, mu)
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
    data must be in df['...'] format.
    :param x:
    :param y:
    :return:
    """
    plt.errorbar(x, y1, yerr=y1_error, xerr=x_error, capsize=5, fmt='.', color="#0000ff", ms=5)
    #plt.plot(x, 0 * x + mu, '-', color="#ff0000", ms=5) # Creates a line at (literature) y value
    plt.xlabel('x')
    plt.ylabel('y')
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
    plt.xlabel('x')
    plt.ylabel('y')
    create_layout()
    plt.show()


print('x_bar: %s'% x_bar)
print('s: %s'% s)
print('delta_x: %s'% delta_x)
t_test(sample, mu)

#print(stats.norm.ppf(1.96))
#print(stats.norm.ppf(1.96))
#histogram(sample, 20)
#create_plot(df['U'], df['delta U'], df['mgd'], df['delta mgd'])
#linear_regression_plot(df['U'], df['delta U'], df['mgd'], df['delta mgd'])

"""
TODO:
chee test bij lin regressie toevoegen
"""