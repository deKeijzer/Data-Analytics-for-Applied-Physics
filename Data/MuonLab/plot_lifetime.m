load 10/lifetime_raw.txt
index = lifetime_raw(:,1)
time = lifetime_raw(:,2);
error = lifetime_raw(:,3);

% Data selectie
% min_val = 1;
% max_val = 650;
% time = time(min_val:max_val);
% index = index(min_val:max_val);
% error = error(min_val:max_val);


% labels & legend zelf aanpassen in createFit
createFit(time, index, error);
hold on

% Correcte significantie maken voor plot 1
xtickformat('%.3f')
ytickformat('%.0f')

% Punt naar comma veranderen voor de de assen van plot 1
x = get(gca, 'XTickLabel');
nieuw_x = strrep(x(:),'.',',');
set(gca, 'XTickLabel', nieuw_x)
% y = get(gca, 'YTickLabel');
% nieuw_y = strrep(y(:),'.',',');
% set(gca, 'YTickLabel', nieuw_y)

% Correcte plot layout maken
xlabel(' Verblijftijd [$\mathrm{\mu}$s] ', 'Interpreter', 'latex')
ylabel(' Meting nummer [-] ', 'Interpreter', 'latex')

