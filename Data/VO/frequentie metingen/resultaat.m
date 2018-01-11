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
[fitresult, gof] = fit( xData, yData, ft );

% Plot fit with data.
h = plot( fitresult, xData, yData, 'v' );
hold on
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
legend_fit_laaghoog = 'Fit oplopend: ax^3+bx^2+cx+d      a=5,109\times10^{-3}     b=-1,493\times10^{-1}     c=1,452   d=-4,689    R^2=0,9971'
legend_fit_hooglaag = 'Fit aflopend: ax^3+bx^2+cx+d      a=1,880\times10^{-2}     b=-5,713\times10^{-1}     c=5,786     d=-19,53   R^2=0,9790'
legend('Meetwaarden oplopend', legend_fit_laaghoog, 'Meetwaarden aflopend',legend_fit_hooglaag, 'Interpreter', 'latex', 'Location' ,'NorthWest')
% legend('Meetwaarden oplopend','Fit oplopend','Meetwaarden aflopend', 'Fit aflopend', 'Interpreter', 'latex', 'Location' ,'NorthWest')

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


