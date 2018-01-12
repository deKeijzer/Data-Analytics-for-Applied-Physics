sample = ['Accelerometer','Luchtdruk']
sample_val = [1, 2];
hoogte = [2.161, 2.1969811];
onzekerheid = [0.3, 0.0000022];

% labels & legend zelf aanpassen in createFit

%plot(sample,hoogte, '*');
errorbar(sample_val(1), hoogte(1), onzekerheid(1), 'vertical', '>', 'Color', 'r');
hold on
errorbar(sample_val(2), hoogte(2), onzekerheid(2), 'vertical', '^', 'Color', 'r');
hold on

legend('Gemeten \tau_0', 'Literatuurwaarde \tau_0', 'Interpreter', 'latex', 'Location' ,'NorthEast')


xlim([-1 3])
% ylim([1.9 2.4])

% Correcte significantie maken voor plot 1
xtickformat('%.1f')
ytickformat('%.3f')

% Punt naar comma veranderen voor de de assen van plot 1
x = get(gca, 'XTickLabel');
nieuw_x = strrep(x(:),'.',',');
set(gca, 'XTickLabel', nieuw_x)
y = get(gca, 'YTickLabel');
nieuw_y = strrep(y(:),'.',',');
set(gca, 'YTickLabel', nieuw_y)

% Legend
%figure( 'Name', 'untitled fit 1', 'Location', 'NorthEast');

% Label axes
ylabel('Levensduur in rust $\tau_0$ [$\mu$s]', 'Interpreter', 'latex')
xlabel('Meting soort [-]', 'Interpreter', 'latex')
grid on


