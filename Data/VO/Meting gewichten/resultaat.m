sample = importdata('resultaat.txt');
sample = sample.data;
time_raw = sample(:,5);

% meting = sample(:,1);
massa = sample(:,1);
freq_piek1 = sample(:,2);
spanning_piek1 = sample(:,3);
freq_piek2 = sample(:,4);
spanning_piek2 = sample(:,5);

fit_piek2(massa, spanning_piek2);


% labels & legend zelf aanpassen in createFit

legend('Meetwaarden', 'Fit: ax+b      a=8,251\times10^{-5}     b=2,131\times10^{-6}     R^2=0.8161', 'Interpreter', 'latex', 'Location' ,'NorthWest')

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


