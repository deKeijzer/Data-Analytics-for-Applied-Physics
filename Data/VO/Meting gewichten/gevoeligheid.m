sample = importdata('resultaat.txt');
sample = sample.data;
time_raw = sample(:,5);

% meting = sample(:,1);
massa = sample(:,1);
freq_piek1 = sample(:,2);
spanning_piek1 = sample(:,3);
freq_piek2 = sample(:,4);
spanning_piek2 = sample(:,5);

% Gegegevens voor bepalen gevoeligheid
rust_massa = 0.25827;
omega = freq_piek2*2*pi;
r = 6.9E-2; % 5.9 +- 0.5 + 1 cm? voor radius vd pijp waar die aan zit
m_tot = massa + rust_massa;

% kracht = omega.^2*r.*m_tot;
a_z = -r*omega
kracht = m_tot.*(a_z+9.81)



% fit_piek1(massa, spanning_piek1);
fit_gevoeligheid_piek1(kracht, spanning_piek1)

% labels & legend zelf aanpassen in createFit

legend('Meetwaarden', 'Fit: ax+b      a=1,466\times10^{-5}     b=-9,546\times10^{-6}     R^2=0,8802', 'Interpreter', 'latex', 'Location' ,'NorthWest')

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
ylabel('Spanning $U$ $\times$ 10$^{-5}$ [V] ', 'Interpreter', 'latex')
grid on


