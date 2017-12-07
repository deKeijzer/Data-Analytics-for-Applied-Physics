function [fitresult, gof] = createFit1(time, index)
%CREATEFIT1(TIME,INDEX)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : time
%      Y Output: index
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  See also FIT, CFIT, SFIT.

%  Auto-generated by MATLAB on 07-Dec-2017 13:49:32


%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( time, index );

% Set up fittype and options.
ft = fittype( 'a*exp(-x/b)+c', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [50 1e-05 -50];
opts.StartPoint = [-5 0 -50];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, xData, yData );
sample_omschrijving = sprintf('Sample 6: N_0 = 1155     \\tau = 0,03495    c = 3,513')

legend( h, sample_omschrijving, 'Fit', 'Location', 'NorthEast' );
% Label axes
xlabel(' Verblijftijd [$\mathrm{\mu}$s] ', 'Interpreter', 'latex')
ylabel(' Meting nummer [-] ', 'Interpreter', 'latex')
grid on


