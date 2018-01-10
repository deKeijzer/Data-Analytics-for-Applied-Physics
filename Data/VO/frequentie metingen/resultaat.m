sample = importdata('resultaat.txt');
sample = sample.data;

% meting = sample(:,1);
meting = sample(:,1);
aandrijf_spanning = sample(:,2);
frequentie = sample(:,3);
gemeten_spanning = sample(:,4);

% Gegegevens voor bepalen gevoeligheid
massa = 0.949802;
rust_massa = 0.06936;
omega = frequentie*2*pi;
r = 6.9E-2; % 5.9 +- 0.5 + 1 cm? voor radius vd pijp waar die aan zit
m_tot = massa + rust_massa;

% kracht = omega.^2*r.*m_tot;
a_z = -r*omega;
kracht = m_tot.*(a_z+9.81);

% Hysteresis zichtbaar, dus arrays splitten voor heen en terugweg

kracht_heen = kracht(1:11);
kracht_terug = kracht(12:24);
gemeten_spanning_heen = gemeten_spanning(1:11);
gemeten_spanning_terug = gemeten_spanning(12:24);
% fit_piek1(massa, spanning_piek1);

% fit_heen(kracht_heen, gemeten_spanning_heen);
% hold on
% fit_terug(kracht_terug, gemeten_spanning_terug);
% hold on

%-----------------------------Start fit heen
%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( kracht_heen, gemeten_spanning_heen );

% Set up fittype and options.
ft = fittype( 'poly3' );

% Fit model to data.
[fitresult1, gof] = fit( xData, yData, ft );

% Plot fit with data.
hold on
h1 = plot( fitresult1, xData, yData, 'v');
%--------------------------------eind fit heen
% -------------------------------Start fit terug
%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( kracht_terug, gemeten_spanning_terug );

% Set up fittype and options.
ft = fittype( 'poly3' );

% Fit model to data.
[fitresult2, gof] = fit( xData, yData, ft );

% Plot fit with data.
h2 = plot( fitresult2, xData, yData, '^')
%--------------------------------eind fit terug
hold on

% labels & legend zelf aanpassen in createFit
legend_fit_laaghoog = 'Fit_{oplopend}: ax^3+bx^2+cx+d      a=-1,236\times10^{-3}     b=3,86\times10^{-2}     c=-3,992\times10^{-1}     d=1,37    R^2=0,9959'
legend_fit_hooglaag = 'Fit_{aflopend}: ax^3+bx^2+cx+d      a=-4,551\times10^{-3}     b=1,337\times10^{-1}     c=-1,308     d=4,269    R^2=0,9842'
legend('Meetwaarden', legend_fit_laaghoog, legend_fit_hooglaag, 'Interpreter', 'latex', 'Location' ,'NorthEast')
% legend('Meetwaarden oplopend','Fit oplopend','Meetwaarden aflopend', 'Fit aflopend', 'Interpreter', 'latex', 'Location' ,'NorthEast')

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
xlabel('Massa $m$ [Kg]', 'Interpreter', 'latex')
ylabel('Spanning $U$ $\times$ 10$^{-4}$ [V] ', 'Interpreter', 'latex')
grid on


