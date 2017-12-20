sample = importdata('samples_data_geselecteerd.txt');
sample = sample.data;
time_raw = sample(:,5);

[hts,ctrs] = hist(time_raw, 59);
bar(ctrs,hts,'hist') % plot histogram
area = sum(hts) * (ctrs(2)-ctrs(1));
xx = linspace(-20,10); % geef x range aan
hold on; 
plot(xx,area*normpdf(xx,mean(time_raw),std(time_raw)),'r-') % plot gauss met berekende mean en std

% Correcte significantie maken voor plot 1
xtickformat('%.1f')
ytickformat('%.0f')

% Punt naar comma veranderen voor de de assen van plot 1
x2 = get(gca, 'XTickLabel');
nieuw_x2 = strrep(x2(:),'.',',');
set(gca, 'XTickLabel', nieuw_x2)
y2 = get(gca, 'YTickLabel');
nieuw_y2 = strrep(y2(:),'.',',');
set(gca, 'YTickLabel', nieuw_y2)

mu = mean(time_raw)
sigma = std(time_raw)
sample_omschrijving = sprintf('Sample 15: \\mu = %.3f     \\sigma = %.3f', mu, sigma)

% Correcte plot layout maken
ylabel('Frequentie [-]')
xlabel('Tijd [ns]')
legend('show', 'Location', 'NorthEast', sample_omschrijving, 'Gauss fit', 'Interpreter', 'latex');
grid on
hold off



