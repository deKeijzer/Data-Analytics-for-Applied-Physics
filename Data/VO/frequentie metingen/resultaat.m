sample = importdata('resultaat.txt');
sample = sample.data;

meting = sample(:,1);
aandrijf_spanning = sample(:,2);
frequentie = sample(:,3);
gemeten_spanning = sample(:,4);
weights = sample(:,6);

% Gegegevens voor bepalen gevoeligheid
massa = 0.949802;
rust_massa = 0.06936;
omega = frequentie*2*pi;
r = 4.3E-2; % 5.9 +- 0.5 + 1 cm? voor radius vd pijp waar die aan zit
m_tot = massa + rust_massa;

% kracht = omega.^2*r.*m_tot;
a_z = r*omega;
kracht = m_tot*(a_z+9.81);

% Hysteresis zichtbaar, dus arrays splitten voor heen en terugweg

kracht_heen = kracht(1:11);
kracht_terug = kracht(12:24);
gemeten_spanning_heen = gemeten_spanning(1:11);
gemeten_spanning_terug = gemeten_spanning(12:24);
wieghts_heen = weights(1:11);
wieghts_terug = weights(12:24);

figure(1);clf % Geeft fig nmr en cleared hem daarna

% fit_piek1(massa, spanning_piek1);

% fit_heen(kracht_heen, gemeten_spanning_heen);
% hold on
% fit_terug(kracht_terug, gemeten_spanning_terug);
% hold on

%-----------------------------Start fit heen
%% Fit: 'untitled fit 1'.
[xData, yData, weights] = prepareCurveData( kracht_heen, gemeten_spanning_heen, wieghts_heen );

% Set up fittype and options.
ft = fittype( 'poly1' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Weights = weights;

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
h1 = plot( fitresult, xData, yData ,'v');
hold on
%--------------------------------eind fit heen
% -------------------------------Start fit terug
%% Fit: 'untitled fit 1'.
[xData, yData, weights] = prepareCurveData( kracht_terug, gemeten_spanning_terug, wieghts_terug );

% Set up fittype and options.
ft = fittype( 'poly1' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Weights = weights;

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
h2 = plot(fitresult, xData, yData,'^');
%--------------------------------eind fit terug
hold on

ylim([0 17E-4])
% labels & legend zelf aanpassen in createFit
legend_fit_laaghoog = 'Fit oplopend: ax+b     a=3,294\times10^{-3}     b=-3,402\times10^{-2}     R^2=0,9973'
legend_fit_hooglaag = 'Fit aflopend: ax+b      a=8,265\times10^{-3}     b=-8,581\times10^{-2}     R^2=0,9521'
% legend('Meetwaarden oplopend', legend_fit_laaghoog, 'Meetwaarden aflopend',legend_fit_hooglaag, 'Interpreter', 'latex', 'Location' ,'NorthWest')
legend('Meetwaarden oplopend','Fit oplopend','Meetwaarden aflopend', 'Fit aflopend', 'Interpreter', 'latex', 'Location' ,'NorthWest')

% xlim([0 0.7])
% Correcte significantie maken voor plot 1
xtickformat('%.3f')
ytickformat('%.3f')

% Punt naar comma veranderen voor de de assen van plot 1
x = get(gca, 'XTickLabel');
nieuw_x = strrep(x(:),'.',',');
set(gca, 'XTickLabel', nieuw_x)
y = get(gca, 'YTickLabel');
nieuw_y = strrep(y(:),'.',',');
set(gca, 'YTickLabel', nieuw_y)

% Label axes
xlabel('Kracht $F$ [N]', 'Interpreter', 'latex')
ylabel('Uitgansspanning $U$ $\times$ 10$^{-3}$ [V] ', 'Interpreter', 'latex')
grid on


