load lifetime_raw.txt
time = lifetime_raw(:,2)
index = lifetime_raw(:,1)

% labels & legend zelf aanpassen in createFit
createFit(time, index)

% Correcte significantie maken voor plot 1
xtickformat('%.3f')
ytickformat('%.0f')

% Punt naar comma veranderen voor de de assen van plot 1
x = get(gca, 'XTickLabel');
nieuw_x = strrep(x(:),'.',',');
set(gca, 'XTickLabel', nieuw_x)
y = get(gca, 'YTickLabel');
nieuw_y = strrep(y(:),'.',',');
set(gca, 'YTickLabel', nieuw_y)

% Correcte plot layout maken
xlabel(' Verblijftijd [$\mathrm{\mu}$s] ', 'Interpreter', 'latex')
ylabel(' Meting nummer [-] ', 'Interpreter', 'latex')
legend('show', 'Location', 'NorthEast', 'Meetwaarden', 'Gauss fit');
hold off