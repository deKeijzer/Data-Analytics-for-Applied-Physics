read_dir = 'call\v2.0\';
write_dir = 'call\matlab_data\';
filename = '2.0';
file_extension = '.txt';

% strcat() voegt string bij elkaar toe 'x'+'y' = 'xy'

Data = fileread(strcat(read_dir,filename,file_extension));
Data = strrep(Data, ',', '.');
FID = fopen(strcat(write_dir,filename,file_extension), 'w');
fwrite(FID, Data, 'char');
fclose(FID);

sample = importdata(strcat(write_dir,filename,file_extension));
sample = sample.data;
time = sample(:,1); 
voltage = sample(:,2);

% Fourier
fourier_sample = voltage;

Fs = 1000;                  % Sampling frequency                    
T = 1/Fs;                   % Sampling period       
L = size(fourier_sample,1); % Length of signal
t = (0:L-1)*T;              % Time vector

Y = fft(fourier_sample);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
clf

% Subplot 1
subplot(2,1,1); 
plot(f,P1);
xlim([0.1 3])
grid on
xlabel('f (Hz)')
ylabel('Amplitude [-]')

% Punt naar comma veranderen voor de assen
x1 = get(gca, 'XTickLabel');
new_x1 = strrep(x1(:),'.',',');
set(gca, 'XTickLabel', new_x1)

% [pks,locs] = findpeaks(P1,f, 'MinPeakHeight', 4E-5)
% %text(locs+.02,pks,num2str((1:numel(pks))'))
% text(locs+.02,pks,num2str((1:numel(pks))'))

% [Peak, PeakIdx] = findpeaks(P1, 'MinPeakHeight', 0.8E-4)
% text(f(PeakIdx), Peak, sprintf('Peaks = %.3f', Peak))


% Subplot 2
subplot(2,1,2);
plot(f,P1);
grid on
xlabel('f (Hz)')
ylabel('Amplitude [-]')

% Punt naar comma veranderen voor de assen
x1 = get(gca, 'XTickLabel');
new_x1 = strrep(x1(:),'.',',');
set(gca, 'XTickLabel', new_x1)
y1 = get(gca, 'YTickLabel');
new_y1 = strrep(y1(:),'.',',');
set(gca, 'YTickLabel', new_y1)

% Plot opslaan
print(strcat(read_dir,'plots\',filename,'.png'),'-dpng')